# What the comparisons establish

## Semantics

Hold the prompt tokens, model revision, numerical policy, path identity, and sequence of sampling draws fixed. For a deterministic prefill under those conditions, reuse the prefix state and independently extend a path-local state. Inductively, if the conditional distributions and sampler agree while histories are equal, the same draw selects the same next token. The claim is conditional: shared mutable state, different kernels, different reductions, or different sampler rules can invalidate the premises.

The GPU validator compares serial batch-1 independent execution with batch-1 shared-prefill continuation. Report pathwise and token agreement, and compare logits, total variation and Jensen–Shannon divergence while histories are still coupled. Report a perturbation control alongside the ordinary comparison. No finite set of matching draws proves universal equality or cross-hardware determinism.

The offline fixture uses exact integer probability buckets and immutable tuple state. It checks the structure of the argument, the negative control and the logical count of prefills. It says nothing about GPU cache mutation, floating-point behavior, or runtime performance.

## Performance

Use paired timings for the same configuration and report latency, throughput, memory pressure, and failures. Preserve raw rows. Key by model, workload, trajectory multiplicity and repetition; never infer the number of trajectories from the number of NVML readings. Report near-memory-limit measurements separately from unconstrained ones. A time-saving observation under one runtime policy is not a portable speedup.

Do not substitute the legacy batch-shaped token check in the benchmark for the matched-shape semantic validator.
