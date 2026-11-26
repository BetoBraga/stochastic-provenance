from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata as im
import json
import math
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None
try:
    from huggingface_hub import model_info
except Exception:
    model_info = None

VERSION = "2026-10-07-semantic-validation-v2"
MASTER_SEED = 20261007
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]
FALLBACK_PROMPTS = [
    "A shop sells 12 notebooks for $3 each and gives a $5 discount on the total. What is the final price? Explain briefly.",
    "A train travels 180 kilometers in 3 hours at a constant speed. How far does it travel in 5 hours?",
    "A container holds 48 red balls and 32 blue balls. If 20 balls are removed proportionally, how many red balls remain?",
    "A farmer has 7 rows of 18 trees and plants 14 additional trees. How many trees are there in total?",
]


def now_utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pkgver(name: str) -> Optional[str]:
    try:
        return im.version(name)
    except Exception:
        return None


def model_revision(model_id: str) -> Optional[str]:
    if model_info is None:
        return None
    try:
        return model_info(model_id).sha
    except Exception:
        return None


def make_uniforms(key: str, n: int, steps: int, device: torch.device) -> torch.Tensor:
    seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=16).digest()[:8], "little")
    rng = np.random.Generator(np.random.Philox(seed))
    arr = rng.random((n, steps), dtype=np.float64).astype(np.float32)
    eps = np.finfo(np.float32).eps
    arr = np.clip(arr, eps, 1.0 - eps)
    return torch.from_numpy(arr).to(device)


def sample_from_uniform(logits: torch.Tensor, u: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
    probs = torch.softmax(logits.float() / float(temperature), dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    cdf[:, -1] = 1.0
    pos = torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous(), right=False).squeeze(-1)
    return torch.clamp(pos, max=probs.shape[-1] - 1)


def distribution_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    pa = torch.softmax(af, dim=-1)
    pb = torch.softmax(bf, dim=-1)
    tv = 0.5 * torch.sum(torch.abs(pa - pb), dim=-1)
    # Jensen-Shannon divergence in nats, numerically stabilized.
    m = 0.5 * (pa + pb)
    eps = 1e-30
    js = 0.5 * torch.sum(pa * (torch.log(pa.clamp_min(eps)) - torch.log(m.clamp_min(eps))), dim=-1)
    js += 0.5 * torch.sum(pb * (torch.log(pb.clamp_min(eps)) - torch.log(m.clamp_min(eps))), dim=-1)
    return {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "tv_distance": float(tv.max().item()),
        "js_divergence_nats": float(js.max().item()),
    }


def clone_cache(cache: Any) -> Any:
    # Prefer round-tripping through the legacy tuple representation when supported.
    try:
        if hasattr(cache, "to_legacy_cache"):
            legacy = cache.to_legacy_cache()
            cloned = tuple(
                tuple(x.clone() if torch.is_tensor(x) else copy.deepcopy(x) for x in layer)
                if isinstance(layer, (tuple, list)) else copy.deepcopy(layer)
                for layer in legacy
            )
            cls = type(cache)
            if hasattr(cls, "from_legacy_cache"):
                return cls.from_legacy_cache(cloned)
    except Exception:
        pass
    try:
        return copy.deepcopy(cache)
    except Exception as e:
        raise RuntimeError(f"Cannot clone cache type {type(cache)!r}: {e}") from e


def forward_kwargs(model) -> Dict[str, Any]:
    import inspect
    try:
        sig = inspect.signature(model.forward)
        if "logits_to_keep" in sig.parameters:
            return {"logits_to_keep": 1}
        if "num_logits_to_keep" in sig.parameters:
            return {"num_logits_to_keep": 1}
    except Exception:
        pass
    return {}


def load_prompts(n: int) -> Tuple[List[str], Dict[str, Any]]:
    bundled = Path(__file__).resolve().parent / "publication_prompts.json"
    if bundled.exists():
        try:
            obj = json.loads(bundled.read_text(encoding="utf-8"))
            prompts = [str(x) for x in obj.get("prompts", [])][:n]
            if len(prompts) >= min(n, 4):
                return prompts, {"source": obj.get("source", "bundled_publication_prompts"), "bundle_sha256": sha256_file(bundled)}
        except Exception:
            pass
    if load_dataset is not None:
        try:
            ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
            return [str(x["question"]) for x in ds], {
                "source": "openai/gsm8k", "split": f"test[:{n}]", "fingerprint": getattr(ds, "_fingerprint", None)
            }
        except Exception as e:
            return FALLBACK_PROMPTS[:n], {"source": "embedded_fallback", "reason": repr(e)}
    return FALLBACK_PROMPTS[:n], {"source": "embedded_fallback", "reason": "datasets unavailable"}


def ids_for_prompt(tok, prompt: str, max_len: int = 512) -> torch.Tensor:
    d = tok(prompt, add_special_tokens=True, truncation=True, max_length=max_len, return_tensors="pt")
    return d["input_ids"].cpu()


def set_policy(policy: str) -> Dict[str, Any]:
    if policy == "strict_eager":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            det = True
        except Exception as e:
            det = f"error:{e!r}"
        return {"policy": policy, "attn_implementation": "eager", "deterministic_algorithms": det, "tf32": False}
    # Operational SDPA policy: same batch-1 shapes, but production-oriented attention path.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    return {"policy": policy, "attn_implementation": "sdpa", "deterministic_algorithms": False, "tf32": True}


def load_model(model_id: str, revision: Optional[str], attn_impl: str):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    is_phi = model_id.lower().startswith("microsoft/phi-4")
    major = int(str(transformers.__version__).split(".", 1)[0])
    trust_remote_code = False
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    kwargs = dict(device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation=attn_impl, trust_remote_code=trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype, **kwargs)
    model.eval()
    return tok, model, dtype


def compare_one_trajectory(model, ids_cpu: torch.Tensor, base_logits: torch.Tensor, base_cache: Any,
                           uniforms_row: torch.Tensor, trajectory: int, temperature: float = 0.8) -> Dict[str, Any]:
    device = next(model.parameters()).device
    fkw = forward_kwargs(model)
    ids = ids_cpu.to(device)
    # Reference performs its own batch-1 prompt prefill.
    with torch.inference_mode():
        ref_out = model(input_ids=ids, use_cache=True, return_dict=True, **fkw)
        ref_logits = ref_out.logits[:, -1, :]
        ref_cache = ref_out.past_key_values
        shared_logits = base_logits.clone()
        shared_cache = clone_cache(base_cache)

        ref_tokens: List[int] = []
        shared_tokens: List[int] = []
        step_metrics: List[Dict[str, Any]] = []
        first_divergence: Optional[int] = None
        histories_equal = True

        for t in range(int(uniforms_row.shape[0])):
            metrics = distribution_metrics(ref_logits, shared_logits) if histories_equal else {
                "max_abs_logit_diff": None, "mean_abs_logit_diff": None, "tv_distance": None, "js_divergence_nats": None
            }
            u = uniforms_row[t].reshape(1)
            rt = sample_from_uniform(ref_logits, u, temperature=temperature)
            st = sample_from_uniform(shared_logits, u, temperature=temperature)
            rti, sti = int(rt.item()), int(st.item())
            same = rti == sti
            if not same and first_divergence is None:
                first_divergence = t
            step_metrics.append({"step": t, "histories_equal_before_step": histories_equal, "same_token": same, **metrics})
            ref_tokens.append(rti)
            shared_tokens.append(sti)
            histories_equal = histories_equal and same
            if t + 1 < int(uniforms_row.shape[0]):
                ref_out = model(input_ids=rt.reshape(1, 1), past_key_values=ref_cache, use_cache=True, return_dict=True, **fkw)
                ref_logits = ref_out.logits[:, -1, :]
                ref_cache = ref_out.past_key_values
                sh_out = model(input_ids=st.reshape(1, 1), past_key_values=shared_cache, use_cache=True, return_dict=True, **fkw)
                shared_logits = sh_out.logits[:, -1, :]
                shared_cache = sh_out.past_key_values

    valid = [m for m in step_metrics if m["histories_equal_before_step"]]
    def vmax(k: str):
        vals = [m[k] for m in valid if m[k] is not None]
        return max(vals) if vals else None
    return {
        "trajectory": trajectory,
        "pathwise_match": ref_tokens == shared_tokens,
        "matched_tokens": int(sum(a == b for a, b in zip(ref_tokens, shared_tokens))),
        "total_tokens": len(ref_tokens),
        "first_divergence_step": first_divergence,
        "max_abs_logit_diff_while_coupled": vmax("max_abs_logit_diff"),
        "max_tv_distance_while_coupled": vmax("tv_distance"),
        "max_js_divergence_nats_while_coupled": vmax("js_divergence_nats"),
        "reference_tokens_sha256": sha256_text(json.dumps(ref_tokens)),
        "shared_tokens_sha256": sha256_text(json.dumps(shared_tokens)),
        "step_metrics": step_metrics,
    }


def semantic_prompt(model_id: str, model, tok, prompt: str, prompt_index: int, n: int, steps: int, policy: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    device = next(model.parameters()).device
    ids = ids_for_prompt(tok, prompt, max_len=512)
    fkw = forward_kwargs(model)
    u = make_uniforms(f"semantic-v2|{policy}|{model_id}|{prompt_index}|{sha256_text(prompt)}", n, steps, device)
    with torch.inference_mode():
        shared_prefill = model(input_ids=ids.to(device), use_cache=True, return_dict=True, **fkw)
        base_logits = shared_prefill.logits[:, -1, :]
        base_cache = shared_prefill.past_key_values

    summary_rows, detailed = [], []
    for i in range(n):
        print(f"      trajectory {i+1}/{n}", flush=True)
        d = compare_one_trajectory(model, ids, base_logits, base_cache, u[i], i)
        summary_rows.append({
            "status":"ok", "model_id":model_id, "policy":policy, "prompt_index":prompt_index,
            "prompt_sha256":sha256_text(prompt), "prompt_tokens":int(ids.shape[1]), "n_trajectories":n,
            "max_new_tokens":steps, **{k:v for k,v in d.items() if k != "step_metrics"}
        })
        for m in d["step_metrics"]:
            detailed.append({
                "model_id":model_id, "policy":policy, "prompt_index":prompt_index, "trajectory":i,
                "prompt_sha256":sha256_text(prompt), **m
            })
    return summary_rows, detailed


def run_negative_control(model_id: str, model, tok, prompt: str, n: int = 4, steps: int = 16) -> Dict[str, Any]:
    # Lightweight check: independent Philox rows should not be replaced by one shared row.
    device = next(model.parameters()).device
    ids = ids_for_prompt(tok, prompt, max_len=256)
    fkw = forward_kwargs(model)
    u = make_uniforms(f"negative|{model_id}|{sha256_text(prompt)}", n, steps, device)
    with torch.inference_mode():
        pre = model(input_ids=ids.to(device), use_cache=True, return_dict=True, **fkw)
        base_logits, base_cache = pre.logits[:, -1, :], pre.past_key_values
    def gen(row_u: torch.Tensor) -> List[int]:
        cache = clone_cache(base_cache); logits = base_logits.clone(); out=[]
        with torch.inference_mode():
            for t in range(steps):
                tokid = sample_from_uniform(logits, row_u[t].reshape(1))
                out.append(int(tokid.item()))
                if t+1 < steps:
                    z = model(input_ids=tokid.reshape(1,1), past_key_values=cache, use_cache=True, return_dict=True, **fkw)
                    logits, cache = z.logits[:,-1,:], z.past_key_values
        return out
    independent = [gen(u[i]) for i in range(n)]
    collapsed = [gen(u[0]) for _ in range(n)]
    return {
        "model_id": model_id,
        "independent_unique_trajectories": len({tuple(x) for x in independent}),
        "collapsed_unique_trajectories": len({tuple(x) for x in collapsed}),
        "negative_control_detected": len({tuple(x) for x in collapsed}) < len({tuple(x) for x in independent}),
    }


def make_zip(root: Path, dst: Path):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file(): z.write(p, arcname=str(p.relative_to(root)).replace("\\", "/"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stochastic Provenance semantic validation v2")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--policies", nargs="*", choices=["strict_eager", "operational_sdpa"], default=["strict_eager", "operational_sdpa"])
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--trajectories", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--output-root", default="results_v3")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")

    stamp=time.strftime("%Y%m%d_%H%M%S")
    out=Path(args.output_root).resolve()/f"semantic_v2_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    prompts,pmeta=load_prompts(args.prompts)
    (out/"prompt_source.json").write_text(json.dumps(pmeta,indent=2),encoding="utf-8")
    (out/"prompt_manifest.json").write_text(json.dumps([{"i":i,"sha256":sha256_text(p),"text":p} for i,p in enumerate(prompts)],indent=2),encoding="utf-8")
    env={
        "version":VERSION,"started_utc":now_utc(),"python":sys.version,"platform":platform.platform(),
        "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"transformers":pkgver("transformers"),
        "datasets":pkgver("datasets"),"gpu":torch.cuda.get_device_name(0),"gpu_total_bytes":torch.cuda.get_device_properties(0).total_memory,
        "models":args.models,"policies":args.policies,"prompts":args.prompts,"trajectories":args.trajectories,"tokens":args.tokens,
    }
    (out/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")

    summary_rows=[]; step_rows=[]; model_meta=[]; neg_rows=[]
    failures=[]
    for policy in args.policies:
        pinfo=set_policy(policy)
        for mi, model_id in enumerate(args.models,1):
            print(f"\n[{policy}] Model {mi}/{len(args.models)} {model_id}", flush=True)
            rev=model_revision(model_id)
            try:
                tok,model,dtype=load_model(model_id,rev,pinfo["attn_implementation"])
            except Exception as e:
                failures.append({"model_id":model_id,"policy":policy,"stage":"load","error":repr(e),"traceback":traceback.format_exc()})
                continue
            model_meta.append({"model_id":model_id,"policy":policy,"revision":rev,"dtype":str(dtype),"params":sum(p.numel() for p in model.parameters()),**pinfo})
            try:
                for pi,prompt in enumerate(prompts):
                    print(f"    prompt {pi+1}/{len(prompts)}", flush=True)
                    s,d=semantic_prompt(model_id,model,tok,prompt,pi,args.trajectories,args.tokens,policy)
                    summary_rows.extend(s); step_rows.extend(d)
                    pd.DataFrame(summary_rows).to_csv(out/"trajectory_summary.csv",index=False)
                    pd.DataFrame(step_rows).to_csv(out/"step_metrics.csv",index=False)
                neg_rows.append({"policy":policy, **run_negative_control(model_id,model,tok,prompts[0],min(4,args.trajectories),min(16,args.tokens))})
            except Exception as e:
                failures.append({"model_id":model_id,"policy":policy,"stage":"semantic","error":repr(e),"traceback":traceback.format_exc()})
            finally:
                del model; del tok; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    sdf=pd.DataFrame(summary_rows)
    if not sdf.empty:
        agg=sdf.groupby(["model_id","policy"],dropna=False).agg(
            trajectories=("pathwise_match","size"), pathwise_matches=("pathwise_match","sum"),
            token_matches=("matched_tokens","sum"), total_tokens=("total_tokens","sum"),
            max_abs_logit_diff=("max_abs_logit_diff_while_coupled","max"),
            max_tv_distance=("max_tv_distance_while_coupled","max"),
            max_js_divergence_nats=("max_js_divergence_nats_while_coupled","max"),
        ).reset_index()
        agg["pathwise_match_rate"] = agg.pathwise_matches/agg.trajectories
        agg["token_match_rate"] = agg.token_matches/agg.total_tokens
    else: agg=pd.DataFrame()
    sdf.to_csv(out/"trajectory_summary.csv",index=False)
    pd.DataFrame(step_rows).to_csv(out/"step_metrics.csv",index=False)
    agg.to_csv(out/"semantic_aggregate.csv",index=False)
    pd.DataFrame(neg_rows).to_csv(out/"negative_control.csv",index=False)
    (out/"model_metadata.json").write_text(json.dumps(model_meta,indent=2),encoding="utf-8")
    (out/"failures.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
    status="completed" if not failures else "completed_with_errors"
    final={"status":status,"finished_utc":now_utc(),"aggregate":agg.to_dict(orient="records"),"failures":failures}
    (out/"SUMMARY.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    zip_path=out.parent/f"semantic_validation_v2_{stamp}.zip"
    make_zip(out,zip_path)
    print(f"\nSEMANTIC_RESULT_ZIP={zip_path}")
    return 0 if not failures else 2

if __name__=="__main__":
    raise SystemExit(main())
