#!/bin/bash
# Lowering exploration for simple_attention_elementwise.mlir
# Follows: https://srock.rocks/book/stablehlo-in-hand

set -e

STABLEHLO_OPT=../stablehlo/build/bin/stablehlo-opt
MLIR_FILE=explore/simple_attention_elementwise.mlir

echo "=== baseline parse ==="
$STABLEHLO_OPT $MLIR_FILE

echo "=== canonicalize (no change) ==="
$STABLEHLO_OPT $MLIR_FILE --canonicalize

echo "=== lower to linalg ==="
$STABLEHLO_OPT $MLIR_FILE --stablehlo-legalize-to-linalg

echo "=== fuse without inline (incomplete - call boundary blocks relu) ==="
$STABLEHLO_OPT $MLIR_FILE \
  --stablehlo-legalize-to-linalg \
  --linalg-fuse-elementwise-ops

echo "=== inline + fuse (complete - single linalg.generic) ==="
$STABLEHLO_OPT $MLIR_FILE \
  --stablehlo-legalize-to-linalg \
  --inline \
  --linalg-fuse-elementwise-ops