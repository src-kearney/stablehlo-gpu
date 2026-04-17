#pragma once
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ADT/StringRef.h"

// Runs the named CPU test against a JIT-compiled module.
// Returns 0 on success, 1 on failure.
int runCpuTest(mlir::ExecutionEngine &engine, llvm::StringRef test);
