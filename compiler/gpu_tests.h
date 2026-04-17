#pragma once
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

#ifdef REMORA_CUDA
// Loads PTX modules and launches the named test kernel on the GPU.
// Returns 0 on success, 1 on failure.
int launchOnGpu(const std::vector<std::string> &ptxModules, llvm::StringRef test);
#endif
