#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

struct PipelineOptions {
  bool printAfterAll = false;
  bool noExecute = false;
  std::string dumpCompilationPhasesTo;
  std::optional<std::string> passPipeline;
};

struct PipelineStep {
  std::string name;
  std::function<mlir::LogicalResult(mlir::OpPassManager &, llvm::raw_ostream &)>
      populate;
};

mlir::LogicalResult runPipeline(
    mlir::ModuleOp module, llvm::ArrayRef<PipelineStep> defaultSteps,
    const PipelineOptions &options, llvm::raw_ostream &errorStream);
