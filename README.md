# remora

[Remora](https://scryfall.com/card/ice/87/mystic-remora) is an MLIR compiler that lowers StableHLO programs to heterogeneous GPU targets, statically placing sparse expert subgraphs across NVIDIA and AMD backends from a single IR source.

`StableHLO → Linalg → heterogenous GPU backends`

## Directory structure

```
remora/
├── .env.example          # Template for local build paths (copy to .env)
├── jax/                  # JAX model definitions (export source)
├── mlir/
│   ├── stablehlo/        # StableHLO IR exported from JAX
│   └── linalg/           # Linalg IR lowered from StableHLO
├── compiler/
│   ├── main.cpp          # MLIR/StableHLO parser + pass pipeline entry point
│   ├── CMakeLists.txt    # Build config (reads MLIR_DIR, STABLEHLO_ROOT/BUILD from .env)
│   └── build/            # CMake build output (gitignored)
└── scripts/
    ├── bootstrap.sh      # Build MLIR + stablehlo-opt from source
    ├── build.sh          # Build remora-compiler (requires .env)
    ├── attention_elementwise_lower_to_linalg.sh
    ├── attention_projection_lowered_to_linalg.sh
    └── elementwise-explore.sh
```

## Prerequisites

- `git`
- `cmake` >= 3.20
- `ninja`
- `python3`
- ~30 GB disk space (LLVM build is large)

## Setup

### 1. Build MLIR and stablehlo-opt from source

Run the bootstrap script from anywhere — it resolves all paths relative to the repo root and clones dependencies into `build-deps/` by default.

```bash
scripts/bootstrap.sh
```

To use a custom build directory:

```bash
scripts/bootstrap.sh --build-dir /path/to/your/build-deps
# or
BUILD_DIR=/path/to/your/build-deps scripts/bootstrap.sh
```

When the build finishes, the script prints the paths you'll need for the next step.

### 2. Configure your .env

The build scripts read local paths from a `.env` file (gitignored). Copy the example and fill in the paths printed by `bootstrap.sh`:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
MLIR_DIR=/path/to/build-deps/llvm-build/lib/cmake/mlir
STABLEHLO_ROOT=/path/to/build-deps/stablehlo
STABLEHLO_BUILD=/path/to/build-deps/stablehlo/build
```

If you used the default `build-deps/` location these will be `<repo-root>/build-deps/...`.

### 3. Build remora-compiler

```bash
scripts/build.sh
```

This sources `.env`, runs CMake, and produces `compiler/build/remora-compiler`.

## Usage

### Run remora-compiler

`remora-compiler` parses a StableHLO or Linalg MLIR file, registers all dialects and passes, and prints the parsed module to stdout. It is the entry point for the pass pipeline as lowering passes are added.

```bash
compiler/build/remora-compiler mlir/stablehlo/simple_attention_elementwise.mlir
```


### Lower StableHLO → Linalg

```bash
scripts/attention_elementwise_lower_to_linalg.sh
scripts/attention_projection_lowered_to_linalg.sh
```

Outputs are written to `mlir/linalg/`. These scripts use `stablehlo-opt` directly; set `STABLEHLO_OPT` in your environment if it is not on your PATH:

```bash
export STABLEHLO_OPT=/path/to/build-deps/stablehlo/build/bin/stablehlo-opt
```

### Explore lowering passes interactively

```bash
scripts/elementwise-explore.sh
```

Runs the elementwise attention file through several progressive lowering steps and prints each result to stdout, useful for understanding what each pass does.

### (Optional) Export fresh StableHLO from JAX

```bash
cd jax
pip install -r requirements.txt
python simple_attention_elementwise.py
python simple_attention_projection.py
```

Outputs are written to `mlir/stablehlo/`.
