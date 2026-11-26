# Experimental evidence

## Controlled environment

Windows 11; RTX 5060 Ti, about 15.93 GiB visible; Torch 2.11.0+cu128, CUDA 12.8 and Transformers 5.19.0. Model-revision identifiers recorded in the study: Qwen3-0.6B `c1899de2`, Qwen3-4B `1cfa9a72`, and Phi-4-mini-instruct `cfbefacb` (short identifiers, not immutable install pins). The native Phi3 loader did not execute model repository code.

| Model | Strict eager: paths / token decisions matched | Operational SDPA: paths / token decisions matched |
|---|---:|---:|
| Qwen3-0.6B | 32 / 2,048 | 32 / 2,048 |
| Qwen3-4B | 32 / 2,048 | 32 / 2,048 |
| Phi-4-mini | 32 / 2,048 | 32 / 2,048 |

Each cell has denominators of 32 paths and 2,048 token decisions. Strict eager requests deterministic algorithms and disables TF32. Operational SDPA allows non-deterministic algorithms and enables TF32. Neither is a cross-device determinism contract. Semantic counts are historical summaries; the six full GPU traces are not included in this checkout.

## Estimator and evidence tiers

- **Wall time** is the median of per-repetition `baseline / shared` ratios, not the ratio of two wall-time medians. The 95% percentile bootstrap resamples entire pairs, with 10,000 draws in the archived analysis. These intervals describe repeated timings in a fixed harness, not transfer across systems.
- **Memory** is the median of per-method peak allocated bytes, divided by 2³⁰ for GiB. It is not reserved VRAM, whole-device NVML usage, or an energy measurement.
- **Qwen timing**: the [selected paired records](../evidence/qwen-paired-records.json) preserve only model/workload identity, repetition, wall milliseconds and peak allocated bytes for three cases. `python tools/check_evidence.py` recomputes the point estimates and validates memory and reported precision. One archival NumPy bootstrap audit was run against the corrected timing archive without regenerating measurements. The included offline standard-library checker does not regenerate the original NumPy bootstrap intervals.
- **Phi timing and semantic counts**: the [aggregate evidence](../evidence/summary.json) retains the recorded results and interval bounds. No Phi timing pairs or full semantic traces were available in the selected archive. They are summary evidence, not independently re-aggregated here.

## Boundary regimes

| Case | Prompt/decode | Paths | Median wall time, baseline → shared | Median paired ratio [95% CI] | Peak allocated, baseline → shared |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B, capacity boundary | 1024/16 | 16 | 57.98 → 3.75 s | 15.474× [15.253, 15.919] | 15.74 → 10.68 GiB |
| Phi-4-mini, decode-dominated | 1024/64 | 16 | 15.39 → 26.21 s | 0.587× [0.561, 0.816] | 13.94 → 10.02 GiB |

The near-capacity case can be sensitive to allocator pressure and host/runtime memory behavior; the evidence does not identify a single mechanism. Short natural prompts supplied a separate low-reuse contrast and were near break-even. Neither boundary belongs in an undifferentiated headline speedup.
