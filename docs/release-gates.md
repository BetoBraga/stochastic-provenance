# Distribution checklist

A record of the release checks and the limits of the artifact.

## Evidence and artifact

- [x] Preserve and verify the archived research source hashes.
- [x] Run the zero-dependency reference, including a negative control, on Windows and Linux.
- [x] Recompute three selected Qwen point estimates from their historical paired records; independently check the recorded bootstrap bounds without rerunning model inference.
- [x] Keep full run bundles, weights and identifying machine logs outside the repository. Distinguish aggregate-only Phi and semantic evidence from re-aggregated Qwen results.
- [ ] If exact full GPU replication is advertised, record complete model and dataset revision hashes and publish a locked, validated environment; prevent silent model or prompt fallback. Until then describe the commands as a new run, not an exact historical replication.
- [ ] If external artifact certification is planned, have a second person run the GPU protocol on a clean host; record discrepancies, not just successes.

## Mandatory before public release

- [x] Confirm code and selected evidence can be published. Apache-2.0 covers project code; external model and dataset terms remain unchanged.
- [x] Check author approval and ensure no active submission restricts this release.
- [x] Review the file inventory and history. The released artifact excludes the manuscript, original run bundles, weights, and machine logs.
- [ ] Verify the public artifact anonymously. Only then add the link to the personal profile.

## Build a private candidate

From a committed checkout, run `python tools/build_candidate.py ../stochastic-provenance-candidate.zip`. It exports an allowlist without Git history or ignored local artifacts. Building the ZIP does not publish it and does not change the license. See the distribution checklist in the source repository.
