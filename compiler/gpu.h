#pragma once
#include "pipeline.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

int runGpu(mlir::ModuleOp module, bool launchOnGpu, llvm::StringRef test,
           const PipelineOptions &options);
