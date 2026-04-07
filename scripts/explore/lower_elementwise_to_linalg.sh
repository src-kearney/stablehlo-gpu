#!/bin/bash
# Lower simple_attention_elementwise.mlir from StableHLO to Linalg.
#
# Requires stablehlo-opt to be on PATH or set via STABLEHLO_OPT env var.
# Run scripts/bootstrap.sh first if you haven't built it yet.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"
STABLEHLO_OPT="${STABLEHLO_OPT:-${STABLEHLO_BUILD:+$STABLEHLO_BUILD/bin/stablehlo-opt}}"
STABLEHLO_OPT="${STABLEHLO_OPT:-stablehlo-opt}"

command -v "$STABLEHLO_OPT" >/dev/null 2>&1 || {
  echo "Error: stablehlo-opt not found. Set STABLEHLO_OPT or run scripts/bootstrap.sh first."
  exit 1
}

"$STABLEHLO_OPT" "$REPO_ROOT/mlir/stablehlo/simple_attention_elementwise.mlir" \
  --stablehlo-legalize-to-linalg \
  --inline \
  --linalg-fuse-elementwise-ops \
  "$@"
