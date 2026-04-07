#!/bin/bash
# Lowering exploration for simple_attention_elementwise.mlir
# Follows: https://srock.rocks/book/stablehlo-in-hand
#
# Requires stablehlo-opt to be on PATH or set via STABLEHLO_OPT env var.
# Run scripts/bootstrap.sh first if you haven't built it yet.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"
STABLEHLO_OPT="${STABLEHLO_OPT:-${STABLEHLO_BUILD:+$STABLEHLO_BUILD/bin/stablehlo-opt}}"
STABLEHLO_OPT="${STABLEHLO_OPT:-stablehlo-opt}"
MLIR_FILE="$REPO_ROOT/mlir/stablehlo/simple_attention_elementwise.mlir"

command -v "$STABLEHLO_OPT" >/dev/null 2>&1 || {
  echo "Error: stablehlo-opt not found. Set STABLEHLO_OPT or run scripts/bootstrap.sh first."
  exit 1
}

echo "=== baseline parse ==="
"$STABLEHLO_OPT" "$MLIR_FILE"

echo "=== canonicalize (no change) ==="
"$STABLEHLO_OPT" "$MLIR_FILE" --canonicalize

echo "=== lower to linalg ==="
"$STABLEHLO_OPT" "$MLIR_FILE" --stablehlo-legalize-to-linalg

echo "=== fuse without inline (incomplete - call boundary blocks relu) ==="
"$STABLEHLO_OPT" "$MLIR_FILE" \
  --stablehlo-legalize-to-linalg \
  --linalg-fuse-elementwise-ops

echo "=== inline + fuse (complete - single linalg.generic) ==="
"$STABLEHLO_OPT" "$MLIR_FILE" \
  --stablehlo-legalize-to-linalg \
  --inline \
  --linalg-fuse-elementwise-ops
