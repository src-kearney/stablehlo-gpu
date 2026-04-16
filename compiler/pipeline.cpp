#include "pipeline.h"

#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdio>

namespace {

std::string sanitizeForFilename(llvm::StringRef name) {
  std::string sanitized;
  sanitized.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')
      sanitized.push_back(c);
    else
      sanitized.push_back('-');
  }
  return sanitized.empty() ? "pipeline-step" : sanitized;
}

mlir::LogicalResult writePhaseDir(mlir::ModuleOp module,
                                   llvm::StringRef root, unsigned index,
                                   llvm::StringRef label,
                                   llvm::raw_ostream &errorStream) {
  char prefix[8];
  std::snprintf(prefix, sizeof(prefix), "%02u", index);
  // Twine is LLVM's lazy string concat. Defers allocation until .str().
  std::string dirPath =
      (llvm::Twine(root) + "/" + prefix + "-" + sanitizeForFilename(label))
          .str();

  if (std::error_code ec = llvm::sys::fs::create_directories(dirPath)) {
    errorStream << "Failed to create '" << dirPath << "': " << ec.message()
                << "\n";
    return mlir::failure();
  }

  std::string filePath = dirPath + "/module.mlir";
  std::error_code ec;
  llvm::raw_fd_ostream os(filePath, ec);
  if (ec) {
    errorStream << "Failed to open '" << filePath << "': " << ec.message()
                << "\n";
    return mlir::failure();
  }

  module.print(os);
  os << "\n";
  return mlir::success();
}

mlir::LogicalResult configureDefaultPipeline(
    mlir::PassManager &pm, llvm::ArrayRef<PipelineStep> steps,
    llvm::raw_ostream &errorStream) {
  for (const PipelineStep &step : steps)
    if (mlir::failed(step.populate(pm, errorStream)))
      return mlir::failure();
  return mlir::success();
}

bool isNameChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' ||
         c == '.';
}

// Extracts the pass name from a pipeline fragment like "convert-scf-to-cf{option=1}".
// Reads leading name chars and stops at the first '(' or '{'.
std::string inferStepName(llvm::StringRef pipelineFragment) {
  llvm::StringRef trimmed = pipelineFragment.trim();
  size_t i = 0;
  while (i < trimmed.size() && isNameChar(trimmed[i]))
    ++i;
  return i == 0 ? "pipeline-step" : trimmed.substr(0, i).str();
}

// Splits "builtin.module(pass1,pass2(opt),pass3)" into one PipelineStep per pass for per-step IR capture.
// Tracks nesting depth so commas inside pass options like "pass{key=a,b}" are not treated as separators.
// There's gotta be a better way to write the following. More concisely at least.
mlir::LogicalResult splitTopLevelPipeline(
    llvm::StringRef pipeline, std::vector<PipelineStep> &steps,
    llvm::raw_ostream &errorStream) {
  llvm::StringRef trimmed = pipeline.trim();
  size_t open = trimmed.find('(');
  size_t close = trimmed.rfind(')');
  if (open == llvm::StringRef::npos || close == llvm::StringRef::npos ||
      close <= open) {
    errorStream << "Expected a rooted MLIR pipeline like builtin.module(...)\n";
    return mlir::failure();
  }

  llvm::StringRef anchor = trimmed.substr(0, open).trim();
  llvm::StringRef body = trimmed.slice(open + 1, close);

  int parenDepth = 0;
  int braceDepth = 0;
  int bracketDepth = 0;
  int angleDepth = 0;
  size_t itemStart = 0;

  auto addItem = [&](llvm::StringRef item) {
    llvm::StringRef fragment = item.trim();
    if (fragment.empty())
      return;
    std::string rooted = (anchor + "(" + fragment + ")").str();
    steps.push_back(PipelineStep{
        inferStepName(fragment),
        [rooted](mlir::OpPassManager &pm,
                 llvm::raw_ostream &errs) -> mlir::LogicalResult {
          return mlir::parsePassPipeline(rooted, pm, errs);
        }});
  };

  for (size_t i = 0; i < body.size(); ++i) {
    char c = body[i];
    switch (c) {
    case '(':
      ++parenDepth;
      break;
    case ')':
      --parenDepth;
      break;
    case '{':
      ++braceDepth;
      break;
    case '}':
      --braceDepth;
      break;
    case '[':
      ++bracketDepth;
      break;
    case ']':
      --bracketDepth;
      break;
    case '<':
      ++angleDepth;
      break;
    case '>':
      if (angleDepth > 0)
        --angleDepth;
      break;
    case ',':
      if (parenDepth == 0 && braceDepth == 0 && bracketDepth == 0 &&
          angleDepth == 0) {
        addItem(body.slice(itemStart, i));
        itemStart = i + 1;
      }
      break;
    default:
      break;
    }
  }
  addItem(body.drop_front(itemStart));

  if (steps.empty()) {
    errorStream << "No passes found in pipeline '" << pipeline << "'\n";
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult runSinglePassManager(mlir::ModuleOp module,
                                         mlir::PassManager &pm) {
  return pm.run(module);
}

} // namespace

mlir::LogicalResult runPipeline(
    mlir::ModuleOp module, llvm::ArrayRef<PipelineStep> defaultSteps,
    const PipelineOptions &options, llvm::raw_ostream &errorStream) {
  if (options.dumpCompilationPhasesTo.empty()) {
    mlir::PassManager pm(module.getContext());
    if (options.printAfterAll) {
      pm.enableIRPrinting(
          nullptr, [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true, /*printAfterOnlyOnChange=*/true);
    }

    if (options.passPipeline) {
      if (mlir::failed(
              mlir::parsePassPipeline(*options.passPipeline, pm, errorStream)))
        return mlir::failure();
    } else if (mlir::failed(
                   configureDefaultPipeline(pm, defaultSteps, errorStream))) {
      return mlir::failure();
    }

    return runSinglePassManager(module, pm);
  }

  const std::string &root = options.dumpCompilationPhasesTo;
  if (std::error_code ec = llvm::sys::fs::create_directories(root)) {
    errorStream << "Failed to create '" << root << "': " << ec.message()
                << "\n";
    return mlir::failure();
  }

  std::vector<PipelineStep> steps;
  if (options.passPipeline) {
    if (mlir::failed(
            splitTopLevelPipeline(*options.passPipeline, steps, errorStream)))
      return mlir::failure();
  } else {
    steps.assign(defaultSteps.begin(), defaultSteps.end());
  }

  // Snapshot 00: IR before any passes run.
  if (mlir::failed(writePhaseDir(module, root, 0, "initial", errorStream)))
    return mlir::failure();

  for (size_t i = 0; i < steps.size(); ++i) {
    const PipelineStep &step = steps[i];

    mlir::PassManager pm(module.getContext());
    if (options.printAfterAll) {
      pm.enableIRPrinting(
          nullptr, [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true, /*printAfterOnlyOnChange=*/true);
    }
    if (mlir::failed(step.populate(pm, errorStream)))
      return mlir::failure();
    if (mlir::failed(runSinglePassManager(module, pm)))
      return mlir::failure();

    // Snapshot N+1: IR after this pass, before the next.
    if (mlir::failed(writePhaseDir(module, root,
                                   static_cast<unsigned>(i + 1),
                                   step.name, errorStream)))
      return mlir::failure();
  }

  return mlir::success();
}
