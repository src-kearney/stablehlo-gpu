#pragma once
#include "pipeline.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

int runCpu(mlir::ModuleOp module, llvm::StringRef test,
           const PipelineOptions &options);
