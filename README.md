# Stochastic Provenance

**Share the prefix. Preserve the trajectory.**

A research artifact for testing when language-model inference can reuse deterministic prefix state without changing the stochastic execution it represents. The question is not merely whether two runs generate plausible text. It is whether **the same logical trajectory, given the same sampling events, makes the same token decisions**.

[Run the demo](#run-in-under-a-minute) · [Inspect the evidence](docs/evidence.md) · [Reproduce a result](docs/reproduction.md) · [Contribute](CONTRIBUTING.md)

## 01 / Semantic validation

**192 / 192 trajectories matched. 12,288 / 12,288 token decisions matched.**

The matrix crossed Qwen3-0.6B, Qwen3-4B and Phi-4-mini with strict eager and operational SDPA. Each model–policy cell used four prompts, eight trajectories per prompt and 64 decoding steps. The serial batch-1 reference and the shared-prefill continuation received the same path-local Philox draws. Maximum observed logit difference, total variation and Jensen–Shannon divergence were zero while histories stayed coupled. In all six cells, a negative control that reused one random stream across four paths collapsed four distinct trajectories to one.

This is exact agreement **in the observed matrix**, not a guarantee across hardware, dtypes, backends or inputs. The [method](docs/method.md) states the conditions under which the argument holds.

## 02 / Latency and memory

Representative 1,024-token prefixes on one RTX 5060 Ti (16 GiB). The timing baseline duplicates the prompt across the batch; it is not the serial semantic reference. Each row uses seven paired timing repetitions. A ratio above 1 favors shared prefill.

| Model | Decode tokens | Paths | Paired wall-time ratio [95% CI] | Peak allocated VRAM, baseline → shared |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 16 | 32 | **2.044×** [2.036, 2.049] | 10.69 → 5.70 GiB |
| Qwen3-4B | 16 | 8 | **2.162×** [2.161, 2.165] | 11.62 → 9.09 GiB |
| Phi-4-mini | 16 | 16 | **2.440×** [2.425, 2.443] | 13.94 → 9.89 GiB |
| Phi-4-mini, decode-heavy | 64 | 16 | **0.587×** [0.561, 0.816] | 13.94 → 10.02 GiB |

**A valid rewrite can still be slower.** The decode-heavy Phi case saved 28.2% of peak allocated VRAM and increased median wall time from 15.39 s to 26.21 s. Separately, the Qwen3-4B baseline at 16 paths approached device capacity (15.74 GiB allocated); the [15.474× boundary observation](docs/evidence.md#boundary-regimes) is not a general algorithmic speedup.

These are measurements of a controlled Transformers harness, **not** a serving-engine leaderboard. [SGLang](https://arxiv.org/abs/2312.07104) studies runtime KV reuse and [Hydragen](https://arxiv.org/abs/2402.05099) accelerates shared-prefix attention. Here, the unit of validation is the stochastic trajectory; production scheduling and kernel comparisons are out of scope.

## 03 / Run in under a minute

Python 3.10+. Run from the repository root. No GPU, models, credentials, or dependencies.

```sh
python -m stochastic_provenance
python -m unittest discover -s tests -v
python tools/check_evidence.py
```

The fixture shares one immutable prefix across three branches, checks their token paths against independently prefetched paths, and requires a deliberately perturbed path to fail. It measures logical prefill calls, not GPU latency. The evidence checker independently re-aggregates selected historical Qwen timing pairs on CPU; it does not execute any model.

| Component | What to inspect |
|---|---|
| [Semantics](docs/method.md) | Conditional equality, path-local randomness, and the two baselines |
| [Evidence](docs/evidence.md) | Workload, numerical policy, bounds and data availability |
| [GPU harness](research/v3.1/) | Archived benchmark, matched-shape validator, and source hashes |
| [Reproduction](docs/reproduction.md) | Offline checks and the separate GPU procedure |

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Citation metadata](CITATION.cff)

## License

The project code is licensed under [Apache-2.0](LICENSE). This license does not relicense third-party models, datasets, or software.
