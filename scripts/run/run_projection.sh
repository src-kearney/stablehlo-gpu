#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$REPO_ROOT/compiler/build/remora" \
  "$REPO_ROOT/mlir/stablehlo/simple_attention_projection.mlir" \
  --kernel=projection "$@"
