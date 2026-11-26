"""Re-aggregate selected historical timing pairs; no model or GPU needed."""
import json
from math import isfinite
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1] / "evidence"


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def check():
    data = read("summary.json")
    records = read("qwen-paired-records.json")
    semantic = data["semantics"]
    paths = (
        len(semantic["models"]) * len(semantic["policies"])
        * semantic["prompts_per_cell"] * semantic["paths_per_prompt"]
    )
    if paths != semantic["matched_paths"] or paths * semantic["steps_per_path"] != semantic["matched_decisions"]:
        raise ValueError("semantic denominators do not reconcile")
    rows = {row["id"]: row for row in data["performance"]}
    observed_keys = {k for k, v in rows.items() if v["evidence"] == "paired_records"}
    if set(records) != observed_keys:
        raise ValueError("unpaired evidence coverage")
    for key, case in records.items():
        row = rows[key]
        if case["workload"] != [row["model"], row["prompt_tokens"], row["decode_tokens"], row["trajectories"]]:
            raise ValueError(f"{key}: workload does not match summary")
        pairs = case["pairs"]
        if len(pairs) != 7 or len({item[0] for item in pairs}) != 7:
            raise ValueError(f"{key}: expected seven distinct repetitions")
        _, bs, ss, bm, sm = zip(*pairs)
        if not all(isfinite(v) and v > 0 for values in (bs, ss, bm, sm) for v in values):
            raise ValueError(f"{key}: invalid latency or allocated bytes")
        expected = {
            "baseline_seconds": (median(bs) / 1000, 2),
            "shared_seconds": (median(ss) / 1000, 2),
            "ratio": (median(b / s for b, s in zip(bs, ss)), 3),
            "baseline_gib": (median(bm) / 2**30, 2),
            "shared_gib": (median(sm) / 2**30, 2),
            "reduction_pct": (100 * (1 - median(sm) / median(bm)), 1),
        }
        for field, (value, precision) in expected.items():
            if round(value, precision) != row[field]:
                raise ValueError(f"{key}: {field} disagrees at reported precision")
        if not row["ci95_low"] <= row["ratio"] <= row["ci95_high"]:
            raise ValueError(f"{key}: point estimate outside recorded interval")
        print(f"{key}: {len(pairs)} pairs, ratio={row['ratio']:.3f}")


if __name__ == "__main__":
    check()
