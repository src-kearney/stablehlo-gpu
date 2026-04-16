#include "cpu.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
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
// CPU kernel runners
// ---------------------------------------------------------------------------

// Inputs: x = all 1.0, bias = all -0.5
// Expected: relu(1.0 + (-0.5)) = relu(0.5) = 0.5 everywhere
static int runElementwise(mlir::ExecutionEngine &engine) {
  const int64_t N = 1, T = 512, D = 768;
  std::vector<float> x_data(N * T * D, 1.0f);
  std::vector<float> bias_data(D, -0.5f);

  StridedMemRefType<float, 3> x_desc;
  x_desc.basePtr = x_desc.data = x_data.data();
  x_desc.offset = 0;
  x_desc.sizes[0] = N; x_desc.sizes[1] = T; x_desc.sizes[2] = D;
  x_desc.strides[0] = T * D; x_desc.strides[1] = D; x_desc.strides[2] = 1;

  StridedMemRefType<float, 1> bias_desc;
  bias_desc.basePtr = bias_desc.data = bias_data.data();
  bias_desc.offset = 0;
  bias_desc.sizes[0] = D;
  bias_desc.strides[0] = 1;

  StridedMemRefType<float, 3> result;

  auto sym = engine.lookup("_mlir_ciface_main");
  if (!sym) {
    llvm::handleAllErrors(sym.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  auto *fn = reinterpret_cast<void (*)(void *, void *, void *)>(*sym);
  fn(&result, &x_desc, &bias_desc);

  llvm::outs() << "result[0][0][0] = " << result.data[0] << " (expected 0.5)\n";
  llvm::outs() << "result[0][0][1] = " << result.data[1] << " (expected 0.5)\n";
  free(result.basePtr);
  return 0;
}

// Inputs: x = all 1.0 (1x512x768), w = all 1/768 (768x768)
// Expected: dot_general(x, w) = 768 * (1.0 * 1/768) = 1.0 everywhere
static int runProjection(mlir::ExecutionEngine &engine) {
  const int64_t N = 1, T = 512, D = 768;
  std::vector<float> x_data(N * T * D, 1.0f);
  std::vector<float> w_data(D * D, 1.0f / D);

  StridedMemRefType<float, 3> x_desc;
  x_desc.basePtr = x_desc.data = x_data.data();
  x_desc.offset = 0;
  x_desc.sizes[0] = N; x_desc.sizes[1] = T; x_desc.sizes[2] = D;
  x_desc.strides[0] = T * D; x_desc.strides[1] = D; x_desc.strides[2] = 1;

  StridedMemRefType<float, 2> w_desc;
  w_desc.basePtr = w_desc.data = w_data.data();
  w_desc.offset = 0;
  w_desc.sizes[0] = D; w_desc.sizes[1] = D;
  w_desc.strides[0] = D; w_desc.strides[1] = 1;

  StridedMemRefType<float, 3> result;

  auto sym = engine.lookup("_mlir_ciface_main");
  if (!sym) {
    llvm::handleAllErrors(sym.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  auto *fn = reinterpret_cast<void (*)(void *, void *, void *)>(*sym);
  fn(&result, &x_desc, &w_desc);

  llvm::outs() << "result[0][0][0] = " << result.data[0] << " (expected 1.0)\n";
  llvm::outs() << "result[0][0][1] = " << result.data[1] << " (expected 1.0)\n";
  free(result.basePtr);
  return 0;
}

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
  auto &engine = *engineOrErr;

  if (test == "projection")
    return runProjection(*engine.get());
  return runElementwise(*engine.get());
}
