#!/bin/bash
# Bootstrap: builds MLIR and stablehlo-opt from source
# Tested on macOS (Apple Silicon) - March 2026
# LLVM commit pinned to stablehlo's known-good version

set -e

git clone https://github.com/openxla/stablehlo.git
git clone https://github.com/llvm/llvm-project.git

LLVM_COMMIT=$(cat stablehlo/build_tools/llvm_version.txt)
echo "Checking out LLVM at $LLVM_COMMIT"
cd llvm-project && git checkout $LLVM_COMMIT && cd ..

sh stablehlo/build_tools/build_mlir.sh llvm-project llvm-build

cmake -GNinja -B stablehlo/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=llvm-build/lib/cmake/mlir \
  -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build stablehlo/build --target stablehlo-opt