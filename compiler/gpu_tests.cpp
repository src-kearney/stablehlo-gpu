#include "gpu_tests.h"
#include "llvm/Support/raw_ostream.h"

#ifdef REMORA_CUDA
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------------------
// Elementwise: relu(x + bias) on GPU
// See gpu.cpp comments for kernel arg layout details.
// ---------------------------------------------------------------------------

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

  std::vector<float> h_x(N * T * D, 1.0f);
  std::vector<float> h_bias(D, -0.5f);
  CUDA_CHECK(cuMemcpyHtoD(d_x,    h_x.data(),    h_x.size()    * sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(d_bias, h_bias.data(), h_bias.size() * sizeof(float)));

  int64_t idx_stride = 1, idx_offset = 0;
  uint64_t x_alloc = d_x, x_align = d_x;
  int64_t  x_off = 0, x_s0 = N, x_s1 = T, x_s2 = D;
  int64_t  x_st0 = T*D, x_st1 = D, x_st2 = 1;
  uint64_t b_alloc = d_bias, b_align = d_bias;
  int64_t  b_off = 0, b_s0 = D, b_st0 = 1;
  float    zero = 0.0f;
  uint64_t r_alloc = d_result, r_align = d_result;
  int64_t  r_off = 0, r_s0 = N, r_s1 = T, r_s2 = D;
  int64_t  r_st0 = T*D, r_st1 = D, r_st2 = 1;

  void *params[] = {
    &idx_stride, &idx_offset,
    &x_alloc, &x_align, &x_off, &x_s0, &x_s1, &x_s2, &x_st0, &x_st1, &x_st2,
    &b_alloc, &b_align, &b_off, &b_s0, &b_st0,
    &zero,
    &r_alloc, &r_align, &r_off, &r_s0, &r_s1, &r_s2, &r_st0, &r_st1, &r_st2,
  };

  CUDA_CHECK(cuLaunchKernel(fn,
    (unsigned)N, (unsigned)T, (unsigned)D,
    1, 1, 1, 0, nullptr, params, nullptr));
  CUDA_CHECK(cuCtxSynchronize());

  std::vector<float> h_result(N * T * D);
  CUDA_CHECK(cuMemcpyDtoH(h_result.data(), d_result, h_result.size() * sizeof(float)));

  llvm::outs() << "GPU result[0][0][0] = " << h_result[0] << " (expected 0.5)\n";
  llvm::outs() << "GPU result[0][0][1] = " << h_result[1] << " (expected 0.5)\n";

  bool allOk = true;
  for (float v : h_result)
    if (std::abs(v - 0.5f) > 1e-5f) { allOk = false; break; }
  llvm::outs() << "Validation: " << (allOk ? "PASS" : "FAIL") << "\n";

  cuMemFree(d_result); cuMemFree(d_x); cuMemFree(d_bias);
  return 0;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

int launchOnGpu(const std::vector<std::string> &ptxModules, llvm::StringRef test) {
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
    llvm::errs() << "--run-gpu for test '" << test << "' not yet implemented\n";
    ret = 1;
  }

  cuCtxDestroy(ctx);
  return ret;
}

#endif // REMORA_CUDA
