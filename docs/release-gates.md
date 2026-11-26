# Distribution checklist

Do not change repository visibility or announce the project until the release owner has cleared the publication checklist. The current distribution is private.

## Evidence and artifact

- [x] Preserve and verify the archived research source hashes.
- [x] Run the zero-dependency reference, including a negative control, on Windows and Linux.
- [x] Recompute three selected Qwen point estimates from their historical paired records; independently check the recorded bootstrap bounds without rerunning model inference.
- [x] Keep full run bundles, weights and identifying machine logs outside the repository. Distinguish aggregate-only Phi and semantic evidence from re-aggregated Qwen results.
- [ ] If exact full GPU replication is advertised, record complete model and dataset revision hashes and publish a locked, validated environment; prevent silent model or prompt fallback. Until then describe the commands as a new run, not an exact historical replication.
- [ ] If external artifact certification is planned, have a second person run the GPU protocol on a clean host; record discrepancies, not just successes.

## Mandatory before public release

- [ ] Confirm license choice, rights holders, scope of data/code redistribution and any required attribution. See the [license candidate](license-review.md).
- [ ] Verify publication and review status. If double-anonymous review applies, do not identify the work in the GitHub profile until the venue rules and author approval permit it. Any reviewer artifact must be a separate approved anonymized copy.
- [ ] Review the release contents and history for secrets, private identifiers, unpublished files and redistribution restrictions. Finalize citation and attribution. Produce a clean export if the private history should not be exposed.
- [ ] Verify the public artifact anonymously. Only then add the link to the personal profile.

## Build a private candidate

From a committed checkout, run `python tools/build_candidate.py ../stochastic-provenance-candidate.zip`. It exports an allowlist without Git history or ignored local artifacts. The resulting ZIP is for private review, not a public or licensed release. See the distribution checklist in the source repository.
