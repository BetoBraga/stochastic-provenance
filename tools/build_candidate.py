"""Create an allowlisted, local-only ZIP from committed files."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = (
    "README.md", "LICENSE", "CITATION.cff", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md",
    ".gitattributes", ".gitignore", ".github", "stochastic_provenance", "tests",
    "docs/method.md", "docs/reproduction.md", "docs/evidence.md",
    "research/v3.1", "evidence", "tools/check_evidence.py", "tools/verify_sources.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output ZIP outside the source tree")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".zip" or output.is_relative_to(ROOT) or output.exists():
        parser.error("use a new .zip path outside the repository")
    subprocess.run(
        ["git", "archive", "--format=zip", "--prefix=stochastic-provenance/",
         "-o", str(output), "HEAD", "--", *ALLOW],
        cwd=ROOT, check=True,
    )
    print(f"Created {output} for private review only; this does not publish the repository.")


if __name__ == "__main__":
    main()
