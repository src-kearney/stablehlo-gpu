#!/bin/bash
./build/bin/stablehlo-opt ~/github/stablehlo-gpu/attention-jax-export/simple_attention_projection.mlir \
  --stablehlo-legalize-to-linalg \
  --inline \
  --linalg-fuse-elementwise-ops > ~/github/stablehlo-gpu/scripts/attention_projection_lowered_to_linalg.mlir