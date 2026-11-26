from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

MASTER_SEED = 20261006
VERSION = "2026-10-07-reaggregate-v3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def paired_bootstrap_median_ratio(a: Sequence[float], b: Sequence[float], seed: int = MASTER_SEED, reps: int = 10000) -> Dict[str, Any]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    ok = np.isfinite(aa) & np.isfinite(bb) & (bb > 0)
    aa, bb = aa[ok], bb[ok]
    if len(aa) == 0:
        return {"median_ratio": None, "ci95_low": None, "ci95_high": None, "paired_n": 0}
    ratios = aa / bb
    med = float(np.median(ratios))
    if len(ratios) == 1:
        return {"median_ratio": med, "ci95_low": med, "ci95_high": med, "paired_n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), size=(reps, len(ratios)))
    boots = np.median(ratios[idx], axis=1)
    return {
        "median_ratio": med,
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "paired_n": int(len(ratios)),
    }


def parse_n_from_benchmark_id(x: Any) -> float:
    if not isinstance(x, str):
        return math.nan
    m = re.search(r"_n(\d+)$", x)
    return float(m.group(1)) if m else math.nan


def canonicalize_transformer_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # In v2 the NVML dictionary overwrote sample_count. Preserve it explicitly.
    if "sample_count" in out.columns:
        out = out.rename(columns={"sample_count": "v2_overwritten_sample_count"})
    if "n_trajectories" in out.columns:
        n = pd.to_numeric(out["n_trajectories"], errors="coerce")
    else:
        n = pd.Series(np.nan, index=out.index, dtype=float)
    parsed = out.get("benchmark_id", pd.Series(index=out.index, dtype=object)).map(parse_n_from_benchmark_id)
    n = n.fillna(parsed)
    out["n_trajectories"] = n.astype("Int64")
    # In successful v2 timing rows, the overwritten sample_count is the number of NVML samples.
    if "v2_overwritten_sample_count" in out.columns:
        out["nvml_sample_count"] = pd.to_numeric(out["v2_overwritten_sample_count"], errors="coerce").astype("Int64")
    return out


def aggregate_transformer(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    ok = raw[raw.get("status", "") == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["model_id", "benchmark_id", "prompt_tokens", "n_trajectories", "max_new_tokens"]
    rows = []
    for key, g in ok.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        for metric in [
            "wall_ms", "gpu_total_ms", "gpu_prefill_ms", "gpu_fork_ms", "gpu_decode_ms",
            "tokens_per_s_wall", "tokens_per_s_gpu", "peak_allocated_bytes", "peak_reserved_bytes",
            "energy_j_est", "avg_power_w", "max_power_w", "max_temp_c", "avg_gpu_util_pct",
        ]:
            if metric not in g.columns:
                continue
            for method in ["batched_duplicate", "shared_prefill"]:
                vals = pd.to_numeric(g.loc[g["method"] == method, metric], errors="coerce").dropna().to_numpy()
                row[f"{method}_{metric}_median"] = float(np.median(vals)) if len(vals) else None
                row[f"{method}_{metric}_iqr"] = float(np.quantile(vals, .75) - np.quantile(vals, .25)) if len(vals) >= 2 else (0.0 if len(vals) else None)
        for metric, prefix in [("wall_ms", "wall_speedup"), ("gpu_total_ms", "gpu_speedup"), ("energy_j_est", "energy_reduction_factor")]:
            if metric not in g.columns:
                continue
            piv = g.pivot_table(index="replicate", columns="method", values=metric, aggfunc="first")
            if {"batched_duplicate", "shared_prefill"}.issubset(piv.columns):
                both = piv[["batched_duplicate", "shared_prefill"]].dropna()
                ci = paired_bootstrap_median_ratio(both["batched_duplicate"].to_numpy(), both["shared_prefill"].to_numpy())
                row.update({f"{prefix}_{k}": v for k, v in ci.items()})
        bmem = row.get("batched_duplicate_peak_allocated_bytes_median")
        smem = row.get("shared_prefill_peak_allocated_bytes_median")
        if bmem and smem and bmem > 0:
            row["peak_allocated_reduction_pct"] = 100.0 * (bmem - smem) / bmem
            row["peak_allocated_saved_gib"] = (bmem - smem) / (1024 ** 3)
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild_legacy_speedups(legacy_dir: Path, out_dir: Path) -> None:
    p = legacy_dir / "legacy_local_timings.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    rows = []
    for (seed, n), g in df.groupby(["seed", "n"]):
        base = g[g.method == "independent"].set_index("replicate")
        for method in ["exact_prefix", "factored_independent", "optimized_exact"]:
            cur = g[g.method == method].set_index("replicate")
            joined = base[["wall_ms"]].join(cur[["wall_ms"]], lsuffix="_base", rsuffix="_method", how="inner")
            ci = paired_bootstrap_median_ratio(joined["wall_ms_base"], joined["wall_ms_method"])
            rows.append({"metric":"wall_ms", "seed":int(seed), "n_trajectories":int(n), "method":method, **ci})
    pd.DataFrame(rows).to_csv(out_dir / "legacy_local_speedups_corrected.csv", index=False)


def make_zip(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p != dst:
                z.write(p, arcname=str(p.relative_to(src)).replace("\\", "/"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_zip", help="Existing stochastic_provenance_results_*.zip from v2")
    ap.add_argument("--output-root", default="results_v3")
    args = ap.parse_args()
    src_zip = Path(args.input_zip).expanduser().resolve()
    if not src_zip.exists():
        raise FileNotFoundError(src_zip)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_root).resolve() / f"reaggregated_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    extract = out / "original_extracted"
    extract.mkdir()
    with zipfile.ZipFile(src_zip, "r") as z:
        z.extractall(extract)

    raw_path = extract / "transformer" / "transformer_raw.csv"
    if not raw_path.exists():
        raise RuntimeError("transformer/transformer_raw.csv not found in input ZIP")
    raw = pd.read_csv(raw_path)
    corrected_raw = canonicalize_transformer_raw(raw)
    corrected_raw.to_csv(out / "transformer_raw_corrected.csv", index=False)
    agg = aggregate_transformer(corrected_raw)
    agg.to_csv(out / "transformer_aggregate_corrected.csv", index=False)

    sem = extract / "transformer" / "semantic_checks.csv"
    if sem.exists():
        shutil.copy2(sem, out / "semantic_checks_v2_original.csv")
    rebuild_legacy_speedups(extract / "legacy", out)

    model_meta = []
    mmp = extract / "transformer" / "model_metadata.json"
    if mmp.exists():
        model_meta = json.loads(mmp.read_text(encoding="utf-8"))
    requested = []
    scp = extract / "study_config.json"
    if scp.exists():
        requested = json.loads(scp.read_text(encoding="utf-8")).get("models", [])
    loaded = [m.get("model_id") for m in model_meta if m.get("status") == "ok"]
    failed = [m for m in model_meta if m.get("status") != "ok"]

    report = {
        "reaggregator_version": VERSION,
        "input_zip": str(src_zip),
        "input_zip_sha256": sha256_file(src_zip),
        "models_requested": requested,
        "models_loaded": loaded,
        "model_failures": failed,
        "corrected_raw_rows": int(len(corrected_raw)),
        "corrected_aggregate_rows": int(len(agg)),
        "notes": [
            "v2 sample_count was overwritten by NVML sample_count; n_trajectories is canonicalized from raw n_trajectories or benchmark_id.",
            "No performance measurement is regenerated; this operation only repairs aggregation/field semantics.",
            "Original extracted files are retained verbatim for auditability.",
        ],
    }
    (out / "REAGGREGATION_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# Reaggregation report",
        "",
        f"Input SHA-256: `{report['input_zip_sha256']}`",
        f"Models requested: {requested}",
        f"Models loaded: {loaded}",
        f"Model failures: {len(failed)}",
        f"Corrected raw rows: {len(corrected_raw)}",
        f"Corrected aggregate rows: {len(agg)}",
        "",
        "This package repairs the v2 field collision only; it does not alter or regenerate benchmark timings.",
    ]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")

    zip_path = out.parent / f"reaggregated_results_{stamp}.zip"
    make_zip(out, zip_path)
    print(f"REAGGREGATED_ZIP={zip_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
