#include "gpu.h"
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

#ifdef REMORA_CUDA
#include <cuda.h>
#endif

#include <cmath>
#include <cstdint>
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
// CUDA harness (only compiled when REMORA_CUDA is defined)
//
// NOTE: UNTESTED. Written on Mac where CUDA is unavailable. Needs validation
// on the RTX 4080. Two assumptions to verify on first run:
//   1. Kernel name: extractKernelName() parses .entry from PTX — check the
//      printed name matches what cuModuleGetFunction expects.
//   2. Grid dim order: assumes innermost linalg loop → blockIdx.x, so for
//      loop order (n,t,d): grid = (D, T, N). Wrong order = garbage output.
// ---------------------------------------------------------------------------

#ifdef REMORA_CUDA

// Matches MLIR's lowered MemRef descriptor layout exactly.
// Fields: allocated ptr, aligned ptr, offset, sizes[Rank], strides[Rank].
// All fields are 8 bytes — no padding on 64-bit targets.
template<int Rank>
struct MemRefDesc {
  uint64_t allocated;
  uint64_t aligned;
  int64_t  offset;
  int64_t  sizes[Rank];
  int64_t  strides[Rank];
};

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      const char *_msg;                                                        \
      cuGetErrorString(_r, &_msg);                                             \
      llvm::errs() << "CUDA error at " #call ": " << _msg << "\n";            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Parse the first kernel name from a PTX string by finding ".entry <name>(".
static std::string extractKernelName(llvm::StringRef ptx) {
  auto pos = ptx.find(".entry ");
  if (pos == llvm::StringRef::npos) return {};
  auto rest = ptx.substr(pos + 7);
  return rest.substr(0, rest.find_first_of("( \t\n")).str();
}

// Launch the elementwise (relu(x + bias)) GPU kernel and validate against
// the expected CPU output.
//
// Grid mapping: linalg parallel dims lower innermost-first to block dims,
// so for loop order (n, t, d): d→blockIdx.x, t→blockIdx.y, n→blockIdx.z.
//
// Kernel args are MemRef descriptors passed by value, matching MLIR's
// lowered struct layout {allocated*, aligned*, offset, sizes[], strides[]}.
static int launchElementwiseOnGpu(CUmodule cuMod, const std::string &ptx) {
  const int64_t N = 1, T = 512, D = 768;

  auto kname = extractKernelName(ptx);
  if (kname.empty()) {
    llvm::errs() << "No .entry found in PTX\n";
    return 1;
  }
  llvm::outs() << "Launching kernel: " << kname << "\n";

  CUfunction fn;
  CUDA_CHECK(cuModuleGetFunction(&fn, cuMod, kname.c_str()));

  CUdeviceptr d_result, d_x, d_bias;
  CUDA_CHECK(cuMemAlloc(&d_result, N * T * D * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_x,      N * T * D * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_bias,   D         * sizeof(float)));

  // x = 1.0, bias = -0.5  →  relu(0.5) = 0.5
  std::vector<float> h_x(N * T * D, 1.0f);
  std::vector<float> h_bias(D, -0.5f);
  CUDA_CHECK(cuMemcpyHtoD(d_x,    h_x.data(),    h_x.size()    * sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(d_bias, h_bias.data(), h_bias.size() * sizeof(float)));

  // gpu.func signature (from --mlir-print-ir-after-all):
  //   %arg0: index          — affine_map stride constant (= 1)
  //   %arg1: index          — affine_map offset constant (= 0)
  //   %arg2: memref<1x512x768xf32, strided<[?,?,?], offset:?>>  — x (dynamic strides)
  //   %arg3: memref<768xf32, strided<[?], offset:?>>            — bias (dynamic)
  //   %arg4: f32            — 0.0 relu threshold
  //   %arg5: memref<1x512x768xf32>                              — result (static strides)
  //
  // Each memref lowers to: [allocated, aligned, offset, sizes..., strides...]
  //   rank 3 → 9 scalars, rank 1 → 5 scalars
  // Total: 1 + 1 + 9 + 5 + 1 + 9 = 26 scalars

  // arg0, arg1: affine map constants captured at kernel outlining time
  int64_t idx_stride = 1, idx_offset = 0;

  // arg2: x memref<1x512x768xf32, strided<[?,?,?], offset:?>>
  uint64_t x_alloc = d_x, x_align = d_x;
  int64_t  x_off = 0, x_s0 = N, x_s1 = T, x_s2 = D;
  int64_t  x_st0 = T*D, x_st1 = D, x_st2 = 1;

  // arg3: bias memref<768xf32, strided<[?], offset:?>>
  uint64_t b_alloc = d_bias, b_align = d_bias;
  int64_t  b_off = 0, b_s0 = D, b_st0 = 1;

  // arg4: f32 relu threshold
  float zero = 0.0f;

  // arg5: result memref<1x512x768xf32> (static layout)
  uint64_t r_alloc = d_result, r_align = d_result;
  int64_t  r_off = 0, r_s0 = N, r_s1 = T, r_s2 = D;
  int64_t  r_st0 = T*D, r_st1 = D, r_st2 = 1;

  void *params[] = {
    &idx_stride, &idx_offset,                                                  // arg0, arg1
    &x_alloc, &x_align, &x_off, &x_s0, &x_s1, &x_s2, &x_st0, &x_st1, &x_st2, // arg2
    &b_alloc, &b_align, &b_off, &b_s0, &b_st0,                                // arg3
    &zero,                                                                     // arg4
    &r_alloc, &r_align, &r_off, &r_s0, &r_s1, &r_s2, &r_st0, &r_st1, &r_st2, // arg5
  }; // 2 + 9 + 5 + 1 + 9 = 26

  CUDA_CHECK(cuLaunchKernel(fn,
    (unsigned)N, (unsigned)T, (unsigned)D,  // gridX, gridY, gridZ — matches gpu.launch order
    1, 1, 1,                                // blockX, blockY, blockZ
    0, nullptr, params, nullptr));
  CUDA_CHECK(cuCtxSynchronize());

  std::vector<float> h_result(N * T * D);
  CUDA_CHECK(cuMemcpyDtoH(h_result.data(), d_result,
                           h_result.size() * sizeof(float)));

  llvm::outs() << "GPU result[0][0][0] = " << h_result[0] << " (expected 0.5)\n";
  llvm::outs() << "GPU result[0][0][1] = " << h_result[1] << " (expected 0.5)\n";

  bool allOk = true;
  for (float v : h_result)
    if (std::abs(v - 0.5f) > 1e-5f) { allOk = false; break; }
  llvm::outs() << "Validation: " << (allOk ? "PASS" : "FAIL") << "\n";

  cuMemFree(d_result); cuMemFree(d_x); cuMemFree(d_bias);
  return 0;
}

static int launchOnGpu(const std::vector<std::string> &ptxModules,
                       llvm::StringRef test) {
  CUDA_CHECK(cuInit(0));
  CUdevice dev; CUDA_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CUDA_CHECK(cuCtxCreate(&ctx, 0, dev));

  int ret = 0;
  if (test == "elementwise") {
    if (ptxModules.size() != 1) {
      llvm::errs() << "Expected 1 PTX module for elementwise, got "
                   << ptxModules.size() << "\n";
      ret = 1;
    } else {
      CUmodule cuMod;
      CUDA_CHECK(cuModuleLoadData(&cuMod, ptxModules[0].c_str()));
      ret = launchElementwiseOnGpu(cuMod, ptxModules[0]);
      cuModuleUnload(cuMod);
    }
  } else {
    llvm::errs() << "--run-gpu for test '" << test
                 << "' not yet implemented\n";
    ret = 1;
  }

  cuCtxDestroy(ctx);
  return ret;
}

#endif // REMORA_CUDA

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
