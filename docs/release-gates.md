# Release gates

**Private by default. Do not change visibility automatically.** This repository is a release candidate scaffold, not an authorized open-source release.

- [ ] Confirm authorship, which owner may publish, and redistribution rights for the study code and any external assets. Select and add LICENSE/NOTICE after approval. Until then no open-source license is granted.
- [ ] Confirm submission status and current venue rules. If an anonymous artifact is needed, create a separate approved export; do not share this repository, its owner name, citation section or commit history with reviewers.
- [ ] Pin the model and dataset revisions and lock the GPU environment. Make the runner fail on unexpected fallback and record full experiment identity.
- [ ] Reproduce one paper table row from raw paired records on a clean environment and check semantic negative control. Separate memory-pressure and unconstrained results. Have a second person reproduce it.
- [ ] Review historical loaders, telemetry and archive handling. Remove host identifiers, paths, prompts not cleared for publication, secrets and unpublished data from all refs intended to be made public.
- [ ] Approve final citation/credits, contributor policy and release scope. Build an allowlisted clean export and inspect it logged out before announcing.
