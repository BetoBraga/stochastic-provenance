# Reproduction

## Offline, verified by the repository test workflow

Run from the repository root:

```sh
python -m stochastic_provenance
python -m unittest discover -s tests -v
python tools/verify_sources.py
```

The source verifier checks byte-exact hashes for the allowlisted study files and parses their Python syntax without importing them. The standard-library smoke tests use no GPU or model dependencies.

## Historical GPU procedure, not validated by CI

The archived dependency files are historical dependency requests, not portable lockfiles. Use an isolated Python 3.13 environment and matching CUDA PyTorch for the historical Windows setup (the archived launcher used Torch 2.11.0+cu128). Verify driver, VRAM and wheel compatibility before installing. Do not install dependencies in the offline environment. The v3.1 recovery notes document the native Phi loader; start with its smoke check before measuring Phi. Do not enable model repository code.

Typical source entry points below are from the root. They can resolve models and datasets over the network; use approved public inputs only. Use a new output directory per run. Model revision resolution and dataset fallback in these scripts are not strict; a run with a different or missing revision or prompt digest is not an exact replication.

```sh
python research/v3.1/semantic_validation_v2.py --models Qwen/Qwen3-0.6B --policies strict_eager --prompts 4 --trajectories 4 --tokens 32 --output-root results-semantic
python research/v3.1/stochastic_provenance_study_v3.py --profile quick --skip-legacy --skip-semantic --output-root results-performance
```

The original prompt bundle, legacy CAM archive, result bundles, and manuscript are not included. The semantic validator records prompt source and can use a dataset or embedded fallback. Use only the reaggregation helper on archives you created or reviewed; its historical ZIP extraction is not hardened for arbitrary input. The study can capture local paths, host metadata, and prompt text in run artifacts. Review and sanitize before sharing. Keep raw outputs off Git.

## For one paper row

First identify the exact table row, source manifest, Git hash, model revision, prompt digest, precision, attention backend, seed, hardware and estimator. Run the semantic and timing checks separately. Preserve failed repetitions and document any environment deviations; compare the intended row to its raw paired records, not an aggregate copied from a report.
