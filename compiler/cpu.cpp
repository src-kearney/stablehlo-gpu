#include "cpu.h"
#include "cpu_tests.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

#include <vector>

namespace {

std::vector<PipelineStep> buildDefaultCpuPipeline() {
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;

  return {
      {"stablehlo-legalize-to-linalg",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
         return mlir::success();
       }},
      {"inline",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createInlinerPass());
         return mlir::success();
       }},
      {"linalg-fuse-elementwise-ops",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
         return mlir::success();
       }},
      {"one-shot-bufferize",
       [bufOpts](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
         return mlir::success();
       }},
      {"convert-linalg-to-loops",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addNestedPass<mlir::func::FuncOp>(
             mlir::createConvertLinalgToLoopsPass());
         return mlir::success();
       }},
      {"lower-affine",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createLowerAffinePass());
         return mlir::success();
       }},
      {"convert-scf-to-cf",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createSCFToControlFlowPass());
         return mlir::success();
       }},
      {"convert-cf-to-llvm",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createConvertControlFlowToLLVMPass());
         return mlir::success();
       }},
      {"convert-arith-to-llvm",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createArithToLLVMConversionPass());
         return mlir::success();
       }},
      {"finalize-memref-to-llvm",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
         return mlir::success();
       }},
      {"convert-func-to-llvm",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createConvertFuncToLLVMPass());
         return mlir::success();
       }},
      {"reconcile-unrealized-casts",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createReconcileUnrealizedCastsPass());
         return mlir::success();
       }},
  };
}

} // namespace

// ---------------------------------------------------------------------------
// CPU backend entry point
// ---------------------------------------------------------------------------

int runCpu(mlir::ModuleOp module, llvm::StringRef test,
           const PipelineOptions &options) {
  // Tag public functions before the pass pipeline so createConvertFuncToLLVMPass
  // generates the C-interface wrapper.
  if (!options.noExecute) {
    module->walk([](mlir::func::FuncOp func) {
      if (func.isPublic())
        func->setAttr("llvm.emit_c_interface",
          mlir::UnitAttr::get(func.getContext()));
    });
  }

  if (mlir::failed(runPipeline(module, buildDefaultCpuPipeline(), options,
                               llvm::errs()))) {
    llvm::errs() << "CPU pass pipeline failed\n";
    return 1;
  }

  if (options.noExecute)
    return 0;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto engineOrErr = mlir::ExecutionEngine::create(module);
  if (!engineOrErr) {
    llvm::handleAllErrors(engineOrErr.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Failed to create ExecutionEngine: " << e.message() << "\n";
    });
    return 1;
  }

  return runCpuTest(*engineOrErr->get(), test);
}
