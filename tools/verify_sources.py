"""Check byte-exact archived files and parse Python without execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "research" / "v3.1"
EXPECTED = {
    "stochastic_provenance_study_v3.py",
    "semantic_validation_v2.py",
    "phi_smoke_test.py",
    "reaggregate_existing_results.py",
    "requirements_qwen_semantic.txt",
    "requirements_phi_compat_4_49.txt",
}


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    if set(entries) != EXPECTED:
        raise SystemExit("incorrect variant archive manifest")
    for name, digest in entries.items():
        payload = (ROOT / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise SystemExit(f"checksum mismatch: {name}")
        if name.endswith(".py"):
            ast.parse(payload.decode("utf-8-sig"), filename=name)
    print(f"Verified {len(entries)} archived files, without executing them.")


if __name__ == "__main__":
    main()
