#include "cpu_tests.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace mlir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static llvm::Expected<void (*)(void *, void *, void *)>
lookupTriple(ExecutionEngine &engine) {
  auto sym = engine.lookup("_mlir_ciface_main");
  if (!sym) return sym.takeError();
  return reinterpret_cast<void (*)(void *, void *, void *)>(*sym);
}

// ---------------------------------------------------------------------------
// Elementwise: relu(x + bias)
// Inputs: x = 1.0, bias = -0.5  →  expected 0.5 everywhere
// ---------------------------------------------------------------------------

static int runElementwise(ExecutionEngine &engine) {
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

  auto fn = lookupTriple(engine);
  if (!fn) {
    llvm::handleAllErrors(fn.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  (*fn)(&result, &x_desc, &bias_desc);

  llvm::outs() << "result[0][0][0] = " << result.data[0] << " (expected 0.5)\n";
  llvm::outs() << "result[0][0][1] = " << result.data[1] << " (expected 0.5)\n";
  free(result.basePtr);
  return 0;
}

// ---------------------------------------------------------------------------
// Projection: matmul(x, w)
// Inputs: x = 1.0, w = 1/768  →  expected 1.0 everywhere
// ---------------------------------------------------------------------------

static int runProjection(ExecutionEngine &engine) {
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

  auto fn = lookupTriple(engine);
  if (!fn) {
    llvm::handleAllErrors(fn.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  (*fn)(&result, &x_desc, &w_desc);

  llvm::outs() << "result[0][0][0] = " << result.data[0] << " (expected 1.0)\n";
  llvm::outs() << "result[0][0][1] = " << result.data[1] << " (expected 1.0)\n";
  free(result.basePtr);
  return 0;
}

// ---------------------------------------------------------------------------
// NER: BERT token classification (bert-base-NER, 9 labels)
// Inputs: [CLS] + 8 dummy tokens + [SEP] + padding, mask covering first 10.
// Validates output shape [1, 128, 9] and finite logits. No tokenizer needed.
// ---------------------------------------------------------------------------

static int runNer(ExecutionEngine &engine) {
  const int64_t B = 1, T = 128, L = 9;

  std::vector<int32_t> ids_data(B * T, 0);
  ids_data[0] = 101;
  for (int i = 1; i <= 8; ++i) ids_data[i] = 1000 + i;
  ids_data[9] = 102;

  std::vector<int32_t> mask_data(B * T, 0);
  for (int i = 0; i <= 9; ++i) mask_data[i] = 1;

  StridedMemRefType<int32_t, 2> ids_desc;
  ids_desc.basePtr = ids_desc.data = ids_data.data();
  ids_desc.offset = 0;
  ids_desc.sizes[0] = B; ids_desc.sizes[1] = T;
  ids_desc.strides[0] = T; ids_desc.strides[1] = 1;

  StridedMemRefType<int32_t, 2> mask_desc;
  mask_desc.basePtr = mask_desc.data = mask_data.data();
  mask_desc.offset = 0;
  mask_desc.sizes[0] = B; mask_desc.sizes[1] = T;
  mask_desc.strides[0] = T; mask_desc.strides[1] = 1;

  StridedMemRefType<float, 3> result;

  auto sym = engine.lookup("_mlir_ciface_main");
  if (!sym) {
    llvm::handleAllErrors(sym.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  auto *fn = reinterpret_cast<void (*)(void *, void *, void *)>(*sym);
  fn(&result, &ids_desc, &mask_desc);

  if (result.sizes[0] != B || result.sizes[1] != T || result.sizes[2] != L) {
    llvm::errs() << "Unexpected output shape: ["
                 << result.sizes[0] << ", " << result.sizes[1] << ", "
                 << result.sizes[2] << "] expected [" << B << ", " << T
                 << ", " << L << "]\n";
    free(result.basePtr);
    return 1;
  }

  static const char *LABELS[] = {
      "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"};
  bool allFinite = true;
  for (int t = 0; t < 10; ++t) {
    int best = 0;
    float bestScore = result.data[t * L];
    for (int l = 1; l < L; ++l) {
      float v = result.data[t * L + l];
      if (!std::isfinite(v)) allFinite = false;
      if (v > bestScore) { bestScore = v; best = l; }
    }
    llvm::outs() << "token[" << t << "] -> " << LABELS[best]
                 << " (score=" << bestScore << ")\n";
  }

  if (!allFinite) {
    llvm::errs() << "Non-finite values in output\n";
    free(result.basePtr);
    return 1;
  }

  llvm::outs() << "output shape: [" << B << ", " << T << ", " << L << "] OK\n";
  free(result.basePtr);
  return 0;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

int runCpuTest(ExecutionEngine &engine, llvm::StringRef test) {
  if (test == "projection") return runProjection(engine);
  if (test == "ner")        return runNer(engine);
  return runElementwise(engine);
}
