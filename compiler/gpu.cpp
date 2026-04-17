#include "gpu.h"
#include "gpu_tests.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include <string>
#include <vector>

namespace {

std::vector<PipelineStep> buildDefaultGpuPipeline() {
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
      {"convert-linalg-to-parallel-loops",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addNestedPass<mlir::func::FuncOp>(
             mlir::createConvertLinalgToParallelLoopsPass());
         return mlir::success();
       }},
      {"gpu-map-parallel-loops",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addNestedPass<mlir::func::FuncOp>(
             mlir::createGpuMapParallelLoopsPass());
         return mlir::success();
       }},
      {"convert-parallel-loops-to-gpu",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addNestedPass<mlir::func::FuncOp>(
             mlir::createConvertParallelLoopToGpuPass());
         return mlir::success();
       }},
      {"gpu-kernel-outlining",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addPass(mlir::createGpuKernelOutliningPass());
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
      {"convert-gpu-ops-to-nvvm",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addNestedPass<mlir::gpu::GPUModuleOp>(
             mlir::createConvertGpuOpsToNVVMOps());
         return mlir::success();
       }},
      {"reconcile-unrealized-casts-gpu-module",
       [](mlir::OpPassManager &pm, llvm::raw_ostream &) {
         pm.addNestedPass<mlir::gpu::GPUModuleOp>(
             mlir::createReconcileUnrealizedCastsPass());
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

// The NVPTX init functions exist in libLLVMNVPTX*.a but are not declared in the
// headers from a host-only llvm-build (Targets.def only lists the native target).
// Forward-declare at file scope so we can call them explicitly.
extern "C" {
  void LLVMInitializeNVPTXTargetInfo();
  void LLVMInitializeNVPTXTarget();
  void LLVMInitializeNVPTXTargetMC();
  void LLVMInitializeNVPTXAsmPrinter();
}

// ---------------------------------------------------------------------------
// GPU backend entry point
// ---------------------------------------------------------------------------

int runGpu(mlir::ModuleOp module, bool launchOnGpuFlag, llvm::StringRef test,
           const PipelineOptions &options) {
  if (mlir::failed(runPipeline(module, buildDefaultGpuPipeline(), options,
                               llvm::errs()))) {
    llvm::errs() << "GPU pass pipeline failed\n";
    return 1;
  }

  if (options.noExecute)
    return 0;

  // Collect all gpu.modules in order — reduction ops (e.g. matmul) generate
  // multiple kernels (fill + compute) that must be emitted and launched in sequence.
  llvm::SmallVector<mlir::gpu::GPUModuleOp> gpuModules;
  module->walk([&](mlir::gpu::GPUModuleOp op) { gpuModules.push_back(op); });
  if (gpuModules.empty()) {
    llvm::errs() << "No gpu.module found after GPU lowering\n";
    return 1;
  }

  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  std::string error;
  llvm::Triple triple("nvptx64-nvidia-cuda");
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "NVPTX target not found: " << error << "\n";
    return 1;
  }

  // sm_89 = Ada Lovelace (RTX 4080)
  llvm::TargetOptions opts;
  auto tm = std::unique_ptr<llvm::TargetMachine>(
    target->createTargetMachine(triple, "sm_89", "+ptx80", opts, std::nullopt)
  );

  std::vector<std::string> ptxModules;
  for (mlir::gpu::GPUModuleOp gpuModule : gpuModules) {
    llvm::LLVMContext llvmCtx;
    auto llvmMod = mlir::translateModuleToLLVMIR(gpuModule, llvmCtx);
    if (!llvmMod) {
      llvm::errs() << "Failed to translate gpu.module to LLVM IR\n";
      return 1;
    }

    llvmMod->setTargetTriple(triple);
    llvmMod->setDataLayout(tm->createDataLayout());

    llvm::SmallString<0> ptxStr;
    llvm::raw_svector_ostream ptxOs(ptxStr);
    llvm::legacy::PassManager codegenPm;
    if (tm->addPassesToEmitFile(codegenPm, ptxOs, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile)) {
      llvm::errs() << "Cannot emit PTX assembly\n";
      return 1;
    }
    codegenPm.run(*llvmMod);
    ptxModules.push_back(ptxStr.str().str());
  }

#ifdef REMORA_CUDA
  if (launchOnGpuFlag)
    return launchOnGpu(ptxModules, test);
#else
  if (launchOnGpuFlag) {
    llvm::errs() << "--run-gpu requires building with CUDA "
                    "(rebuild with -DREMORA_CUDA=ON)\n";
    return 1;
  }
#endif

  for (auto &ptx : ptxModules)
    llvm::outs() << ptx;
  return 0;
}
