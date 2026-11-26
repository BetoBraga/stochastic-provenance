# Stochastic Provenance

**Reuse the prefix. Keep the random choices attributable to each trajectory.**

Generating several continuations from one prompt repeats work before the first sampled token. This project studies when the prefill state can be reused without changing what each continuation samples. It compares a serial independent reference with shared-prefill execution using the same draws for each trajectory, and separates semantic evidence from speed and memory measurements.

**Status:** private, pre-release research artifact. The offline demo is a reference model, not a GPU benchmark or a production inference engine. No preprint, measured speedup, or hardware-equivalence claim is included here.

## Run the offline demo

Python 3.10+. No dependencies, model downloads, credentials, GPU, or network required.

```sh
python -m stochastic_provenance
python -m unittest discover -s tests -v
```

The demo runs three trajectories from the same prefix. The serial reference does three prefills; the shared path does one. The trajectories are compared with the same explicit draws, and a perturbed negative control must disagree. Prefill counts are logical work units, not speedup measurements.

## Explore the research artifact

| Path | Purpose |
| --- | --- |
| `stochastic_provenance/` | Offline, finite-distribution reference and negative control |
| `research/v3.1/` | Curated study sources: CUDA benchmark, matched-shape semantic validator, Phi smoke check and reaggregation |
| [Method and evidence](docs/method.md) | Comparison contract and limits of each claim |
| [Reproduction guide](docs/reproduction.md) | Offline verification, source checks and separate GPU procedure |
| [Release gates](docs/release-gates.md) | Evidence, rights, anonymity, and publication checklist |

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Citation metadata](CITATION.cff)
