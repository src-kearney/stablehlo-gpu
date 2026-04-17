#include "cpu.h"
#include "gpu.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: remora <input.mlir> "
                    "--test=<elementwise|projection> "
                    "[--emit-ptx] [--run-gpu] "
                    "[--mlir-print-ir-after-all] "
                    "[--pass-pipeline='builtin.module(...)'] "
                    "[--dump-compilation-phases-to=<dir>] "
                    "[--no-execute]\n";
    return 1;
  }

  llvm::StringRef test;
  bool emitPtxMode = false;
  bool runGpuMode = false;
  PipelineOptions options;
  for (int i = 2; i < argc; i++) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--mlir-print-ir-after-all")
      options.printAfterAll = true;
    else if (arg == "--emit-ptx")
      emitPtxMode = true;
    else if (arg == "--run-gpu")
      runGpuMode = true;
    else if (arg == "--no-execute")
      options.noExecute = true;
    else if (arg.consume_front("--test="))
      test = arg;
    else if (arg.consume_front("--pass-pipeline="))
      options.passPipeline = arg.str();
    else if (arg.consume_front("--dump-compilation-phases-to="))
      options.dumpCompilationPhasesTo = arg.str();
  }

  if (!options.noExecute && !emitPtxMode && !runGpuMode &&
      test != "elementwise" && test != "projection" && test != "ner") {
    llvm::errs() << "Unknown test '" << test
                 << "'. Use --test=elementwise, --test=projection, or --test=ner\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();
  mlir::registerAllExtensions(registry); // we should probably turn this off / make more finer grainer
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);

  mlir::MLIRContext ctx(registry);
  if (options.printAfterAll || !options.dumpCompilationPhasesTo.empty())
    ctx.disableMultithreading();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);
  if (!module) {
    llvm::errs() << "Failed to parse: " << argv[1] << "\n";
    return 1;
  }

  if (emitPtxMode || runGpuMode)
    return runGpu(*module, runGpuMode, test, options);
  return runCpu(*module, test, options);
}
