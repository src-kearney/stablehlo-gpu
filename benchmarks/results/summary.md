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

## sweep_skew — vLLM vs Remora per-expert dispatch

> Note: vLLM used its default (un-tuned) MoE config. Remora includes Python-level gather/scatter overhead in its timing.

| distribution | vllm_ms | remora_ms | speedup | vllm_tok/s | remora_tok/s | tokens per expert              |
|--------------|--------:|----------:|--------:|-----------:|-------------:|-------------------------------|
| uniform      |   6.749 |     8.785 |  0.77x  |     75,859 |       58,278 | 140 131 132 131 117 125 128 120 |
| zipf_0.5     |   6.691 |     8.630 |  0.78x  |     76,515 |       59,326 | 215 173 129 114 105 109  93  86 |
| zipf_1.0     |   6.973 |     8.803 |  0.79x  |     73,428 |       58,161 | 313 202 142  89  83  68  70  57 |
| zipf_1.5     |   6.947 |     8.952 |  0.78x  |     73,698 |       57,192 | 420 220 127  77  71  53  31  25 |
| zipf_2.0     |   6.830 |     9.266 |  0.74x  |     74,959 |       55,255 | 462 272 117  56  32  39  32  14 |
| single       |   5.096 |     3.055 |  1.67x  |    100,469 |      167,588 | 512 512   0   0   0   0   0   0 |

**Key finding:** Remora wins only at `single` (1.67x), where 6 experts have zero tokens and vLLM still pads
all 8 slots to 512. Under uniform and Zipf distributions, the Python gather/scatter loop adds enough overhead
to negate the per-expert savings — Remora is 0.74–0.79x slower. The crossover point is high skew where
most experts are idle.
