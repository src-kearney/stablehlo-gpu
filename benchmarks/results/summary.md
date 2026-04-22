# Benchmark Results — RTX 4090

Mixtral-8x7B dims: hidden=4096, intermediate=14336, experts=8, top_k=2, batch=512 tokens.

## vLLM fused_moe baseline

> Note: vLLM used its default (un-tuned) MoE config; no autotune JSON found for this GPU.

| distribution | latency (ms) | tokens/sec | tokens per expert              |
|--------------|-------------:|------------|-------------------------------|
| uniform      |        6.970 |     73,457 | 117 134 130 132 134 137 123 117 |
| zipf_0.5     |        6.664 |     76,828 | 193 191 130 119 119 103  89  80 |
| zipf_1.0     |        6.399 |     80,006 | 291 204 153 101  96  63  62  54 |
| zipf_1.5     |        7.104 |     72,068 | 415 210 144  85  65  33  45  27 |
| zipf_2.0     |        7.133 |     71,784 | 465 249 132  75  36  31  23  13 |
| single       |        5.091 |    100,561 | 512 512   0   0   0   0   0   0 |

## Triton per-expert kernel — correctness

Token distribution: `[300, 200, 100, 50, 20, 10, 5, 2]`

| expert | tokens | BLOCK_M | max \|err\|  | mean \|err\|  | status |
|-------:|-------:|--------:|-------------|--------------|--------|
|      0 |    300 |     128 |   6.10e-05  |   3.33e-06   | PASS   |
|      1 |    200 |      64 |   6.10e-05  |   3.31e-06   | PASS   |
|      2 |    100 |      32 |   6.10e-05  |   3.29e-06   | PASS   |
|      3 |     50 |      32 |   6.10e-05  |   3.32e-06   | PASS   |
|      4 |     20 |      16 |   6.10e-05  |   3.28e-06   | PASS   |
|      5 |     10 |      16 |   6.10e-05  |   3.34e-06   | PASS   |
|      6 |      5 |      16 |   6.10e-05  |   3.28e-06   | PASS   |
|      7 |      2 |      16 |   3.05e-05  |   3.28e-06   | PASS   |

All 8 experts pass at atol=0.001 (fp16).

## sweep_skew v1 — boolean mask gather/scatter (baseline, slow)

> Remora used E boolean masks + E fancy-index gathers + E masked index_put scatters per top-k slot.

| distribution | vllm_ms | remora_ms | speedup | vllm_tok/s | remora_tok/s | tokens per expert              |
|--------------|--------:|----------:|--------:|-----------:|-------------:|-------------------------------|
| uniform      |   6.749 |     8.785 |  0.77x  |     75,859 |       58,278 | 140 131 132 131 117 125 128 120 |
| zipf_0.5     |   6.691 |     8.630 |  0.78x  |     76,515 |       59,326 | 215 173 129 114 105 109  93  86 |
| zipf_1.0     |   6.973 |     8.803 |  0.79x  |     73,428 |       58,161 | 313 202 142  89  83  68  70  57 |
| zipf_1.5     |   6.947 |     8.952 |  0.78x  |     73,698 |       57,192 | 420 220 127  77  71  53  31  25 |
| zipf_2.0     |   6.830 |     9.266 |  0.74x  |     74,959 |       55,255 | 462 272 117  56  32  39  32  14 |
| single       |   5.096 |     3.055 |  1.67x  |    100,469 |      167,588 | 512 512   0   0   0   0   0   0 |

## profiler — gather/scatter breakdown before fix (zipf_1.0)

| phase              | calls | cuda_total_ms | cuda_avg_us |
|--------------------|------:|--------------:|------------:|
| scatter            |    80 |       10.895  |      136.2  |
| run_remora_outlined|    10 |       35.281  |     3528.1  |
| gather             |    80 |        2.869  |       35.9  |
| any_check          |   160 |        1.178  |        7.4  |
| mask_creation      |    80 |        0.091  |        1.1  |

Wall time: **9.649 ms/iter** — scatter (2.18 ms) + gather+any_check (1.07 ms) = 33% of wall time.

## profiler — after sort-based vectorized gather/scatter (zipf_1.0)

Replaced E×2 mask/gather/scatter ops with: `argsort → index → split → run → cat → index_add_`.

| phase              | calls | cuda_total_ms | cuda_avg_us |
|--------------------|------:|--------------:|------------:|
| argsort            |    10 |        0.544  |       54.4  |
| bincount           |    10 |        0.600  |       60.0  |
| split              |    10 |        0.008  |        0.8  |
| run_remora_outlined|    10 |       35.224  |     3522.4  |
| cat                |    10 |        0.066  |        6.6  |
| index_add          |    10 |        0.247  |       24.7  |

Wall time: **7.413 ms/iter** — scatter+gather overhead cut from ~3.25 ms to ~1.47 ms (2.2x reduction).

## sweep_skew v2 — sort-based vectorized gather/scatter

> Note: vLLM used its default (un-tuned) MoE config. Remora timing includes argsort + bincount + index_add_ overhead.

| distribution | vllm_ms | remora_ms | speedup | vllm_tok/s | remora_tok/s | tokens per expert              |
|--------------|--------:|----------:|--------:|-----------:|-------------:|-------------------------------|
| uniform      |   6.748 |     7.317 |  0.92x  |     75,878 |       69,975 | 140 131 132 131 117 125 128 120 |
| zipf_0.5     |   6.660 |     7.183 |  0.93x  |     76,876 |       71,279 | 215 173 129 114 105 109  93  86 |
| zipf_1.0     |   6.972 |     7.291 |  0.96x  |     73,432 |       70,225 | 313 202 142  89  83  68  70  57 |
| zipf_1.5     |   6.950 |     7.472 |  0.93x  |     73,665 |       68,527 | 420 220 127  77  71  53  31  25 |
| zipf_2.0     |   6.816 |     7.825 |  0.87x  |     75,120 |       65,429 | 462 272 117  56  32  39  32  14 |
| single       |   5.092 |     2.649 |  1.92x  |    100,554 |      193,282 | 512 512   0   0   0   0   0   0 |

**Key finding:** Vectorizing gather/scatter cut Remora's overhead from ~3.25 ms to ~1.47 ms/iter, closing
the gap significantly (worst case 0.74x → 0.87x; single improved from 1.67x → 1.92x). Remora does not yet
beat vLLM at Zipf distributions — the remaining gap is ~0.3–0.6 ms/iter, attributable to argsort + bincount
(~1.1 ms/iter combined for 2 top-k slots) which a compiled MLIR dispatch would eliminate entirely. The
crossover point is `single`-class skew where most experts are idle — confirmed by sweep_concentration below.

## sweep_concentration — crossover analysis

Sweeps expert-0 concentration from 0.125 (uniform) to 1.0 (single) in 18 steps,
fine-grained in [0.75, 1.0] where the crossover was expected.

| concentration | top_expert slots | vllm_ms | remora_ms | speedup | winner |
|--------------:|-----------------:|--------:|----------:|--------:|--------|
|        0.1250 |              140 |   6.753 |     7.311 |  0.92x  | vllm   |
|        0.2500 |              236 |   6.038 |     7.182 |  0.84x  | vllm   |
|        0.3750 |              319 |   6.576 |     7.374 |  0.89x  | vllm   |
|        0.5000 |              402 |   7.358 |     7.430 |  0.99x  | ~tie   |
|        0.6250 |              445 |   6.807 |     7.726 |  0.88x  | vllm   |
|        0.7000 |              470 |   7.206 |     7.502 |  0.96x  | ~tie   |
|        0.7500 |              493 |   7.060 |     7.632 |  0.93x  | vllm   |
|        0.8000 |              499 |   7.207 |     7.577 |  0.95x  | ~tie   |
|        0.8500 |              497 |   7.191 |     7.683 |  0.94x  | vllm   |
|        0.8750 |              505 |   7.191 |     7.707 |  0.93x  | vllm   |
|        0.9000 |              506 |   7.191 |     7.671 |  0.94x  | vllm   |
|        0.9250 |              510 |   7.215 |     7.820 |  0.92x  | vllm   |
|        0.9375 |              511 |   7.195 |     7.731 |  0.93x  | vllm   |
|        0.9500 |              511 |   7.063 |     7.669 |  0.92x  | vllm   |
|        0.9625 |              512 |   7.189 |     7.785 |  0.92x  | vllm   |
|        0.9750 |              512 |   6.941 |     7.341 |  0.95x  | vllm   |
|        0.9875 |              512 |   6.759 |     7.340 |  0.92x  | vllm   |
|        1.0000 |              512 |   5.106 |     2.624 |  1.95x  | remora |

**Key finding — the crossover is a cliff, not a slope.**
Remora wins *only* at c=1.0 (the true "single" distribution) where 6 of 8 experts receive
exactly 0 tokens and their Triton kernel pairs are skipped entirely. At c=0.9875, all 8
experts still receive at least 1 token so all 16 kernel pairs still fire — and argsort +
bincount overhead (~1.1 ms/iter) is paid regardless of concentration.

The implication: Remora's advantage is gated on **zero-token expert elimination**, not on
load imbalance per se. A compiled MLIR dispatch (from ExpertOutliningPass) that emits static
per-expert control flow would avoid the runtime argsort/bincount entirely and shift the
crossover to lower concentrations. That is the next optimization target.
