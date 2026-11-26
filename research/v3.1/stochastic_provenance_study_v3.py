from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil

try:
    import pynvml
except Exception:
    pynvml = None

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import datasets
    from datasets import load_dataset
except Exception:
    datasets = None
    load_dataset = None

try:
    from huggingface_hub import model_info
except Exception:
    model_info = None


STUDY_VERSION = "2026-10-07-local-transformer-v3"
MASTER_SEED = 20261006

FALLBACK_PROMPTS = [
    "A shop sells 12 notebooks for $3 each and gives a $5 discount on the total. What is the final price? Explain briefly.",
    "A train travels 180 kilometers in 3 hours at a constant speed. How far does it travel in 5 hours?",
    "A container holds 48 red balls and 32 blue balls. If 20 balls are removed proportionally, how many red balls remain?",
    "A farmer has 7 rows of 18 trees and plants 14 additional trees. How many trees are there in total?",
    "A rectangle has length 15 cm and width 8 cm. What are its area and perimeter?",
    "If a server processes 240 requests in 6 minutes at a constant rate, how many requests does it process in 15 minutes?",
    "A budget of $840 is divided equally among 7 teams. Each team then spends $35. How much remains per team?",
    "A car consumes 6 liters of fuel per 100 km. How many liters are needed for a 450 km trip?",
]


@dataclass
class ProfileConfig:
    models: List[str]
    micro_prompt_lengths: List[int]
    micro_decode_lengths: List[int]
    model_sample_counts: Dict[str, List[int]]
    semantic_max_new_tokens: int
    real_max_new_tokens: int
    micro_repetitions: int
    semantic_prompts: int
    real_prompts: int
    real_sample_count: int
    real_repetitions: int
    run_legacy_full: bool


def get_profile(name: str) -> ProfileConfig:
    if name == "quick":
        return ProfileConfig(
            models=["Qwen/Qwen3-0.6B"],
            micro_prompt_lengths=[128, 512],
            micro_decode_lengths=[16],
            model_sample_counts={"Qwen/Qwen3-0.6B": [1, 4, 8]},
            semantic_max_new_tokens=16,
            real_max_new_tokens=16,
            micro_repetitions=2,
            semantic_prompts=2,
            real_prompts=2,
            real_sample_count=4,
            real_repetitions=1,
            run_legacy_full=False,
        )
    if name == "standard":
        return ProfileConfig(
            models=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"],
            micro_prompt_lengths=[128, 512, 1024],
            micro_decode_lengths=[16, 64],
            model_sample_counts={
                "Qwen/Qwen3-0.6B": [1, 4, 8, 16],
                "Qwen/Qwen3-4B": [1, 4, 8],
            },
            semantic_max_new_tokens=32,
            real_max_new_tokens=32,
            micro_repetitions=5,
            semantic_prompts=3,
            real_prompts=3,
            real_sample_count=8,
            real_repetitions=3,
            run_legacy_full=True,
        )
    return ProfileConfig(
        models=[
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-4B",
            "microsoft/Phi-4-mini-instruct",
        ],
        micro_prompt_lengths=[128, 512, 1024],
        micro_decode_lengths=[16, 64],
        model_sample_counts={
            "Qwen/Qwen3-0.6B": [1, 4, 8, 16, 32],
            "Qwen/Qwen3-4B": [1, 4, 8, 16],
            "microsoft/Phi-4-mini-instruct": [1, 4, 8, 16],
        },
        semantic_max_new_tokens=64,
        real_max_new_tokens=64,
        micro_repetitions=7,
        semantic_prompts=4,
        real_prompts=4,
        real_sample_count=8,
        real_repetitions=5,
        run_legacy_full=True,
    )


def now_utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def run_capture(cmd: Sequence[str]) -> Dict[str, Any]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=60, check=False)
        return {"cmd": list(cmd), "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"cmd": list(cmd), "error": repr(e)}


def collect_hardware() -> Dict[str, Any]:
    hw: Dict[str, Any] = {
        "timestamp_utc": now_utc(),
        "study_version": STUDY_VERSION,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "hostname_hash": sha256_text(platform.node())[:16],
        "local_machine_note": "User-owned local host. No cloud/shared-tenant CPU is used by this study; ordinary OS background activity may still exist and is measured before timed runs.",
    }
    if platform.system() == "Windows":
        ps = (
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress"
        )
        hw["windows_cpu_cim"] = run_capture(["powershell.exe", "-NoProfile", "-Command", ps])
        ps_mem = (
            "Get-CimInstance Win32_PhysicalMemory | "
            "Select-Object Manufacturer,PartNumber,Capacity,Speed,ConfiguredClockSpeed | ConvertTo-Json -Compress"
        )
        hw["windows_memory_cim"] = run_capture(["powershell.exe", "-NoProfile", "-Command", ps_mem])
    hw["nvidia_smi"] = run_capture([
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total,power.limit,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        hw["torch_cuda"] = {
            "device_index": idx,
            "device_name": torch.cuda.get_device_name(idx),
            "capability": list(torch.cuda.get_device_capability(idx)),
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return hw


def collect_software() -> Dict[str, Any]:
    names = [
        "torch", "transformers", "accelerate", "huggingface_hub", "datasets",
        "numpy", "pandas", "scipy", "scikit-learn", "nvidia-ml-py", "psutil",
    ]
    return {
        "timestamp_utc": now_utc(),
        "packages": {name: safe_package_version(name) for name in names},
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "datasets": getattr(datasets, "__version__", None),
        "env": {
            k: os.environ.get(k)
            for k in [
                "HF_HOME", "PYTHONHASHSEED", "TOKENIZERS_PARALLELISM",
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
            ]
        },
    }


def set_process_priority() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        p = psutil.Process()
        if platform.system() == "Windows":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            info["priority"] = "HIGH_PRIORITY_CLASS"
        else:
            info["priority"] = p.nice()
        info["cpu_affinity"] = p.cpu_affinity() if hasattr(p, "cpu_affinity") else None
    except Exception as e:
        info["error"] = repr(e)
    return info


class NVMLSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = False
        self.error: Optional[str] = None
        self.handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.available = True
            except Exception as e:
                self.error = repr(e)

    def _read(self) -> Dict[str, float]:
        assert self.handle is not None
        row: Dict[str, float] = {"t": time.perf_counter()}
        try:
            row["power_w"] = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            row["power_w"] = math.nan
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            row["gpu_util_pct"] = float(util.gpu)
            row["mem_util_pct"] = float(util.memory)
        except Exception:
            row["gpu_util_pct"] = math.nan
            row["mem_util_pct"] = math.nan
        try:
            row["temp_c"] = float(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            row["temp_c"] = math.nan
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            row["nvml_mem_used_bytes"] = float(mem.used)
        except Exception:
            row["nvml_mem_used_bytes"] = math.nan
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._read())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        if self.available:
            try:
                self.samples.append(self._read())
            except Exception:
                pass
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if not self.available:
            return {"nvml_available": False, "nvml_error": self.error}
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self.samples.append(self._read())
        except Exception:
            pass
        if not self.samples:
            return {"nvml_available": True, "nvml_sample_count": 0}
        powers = [s["power_w"] for s in self.samples if math.isfinite(s.get("power_w", math.nan))]
        temps = [s["temp_c"] for s in self.samples if math.isfinite(s.get("temp_c", math.nan))]
        utils = [s["gpu_util_pct"] for s in self.samples if math.isfinite(s.get("gpu_util_pct", math.nan))]
        energy_j = 0.0
        valid_pairs = 0
        for a, b in zip(self.samples[:-1], self.samples[1:]):
            pa, pb = a.get("power_w", math.nan), b.get("power_w", math.nan)
            if math.isfinite(pa) and math.isfinite(pb):
                energy_j += 0.5 * (pa + pb) * (b["t"] - a["t"])
                valid_pairs += 1
        return {
            "nvml_available": True,
            "nvml_sample_count": len(self.samples),
            "energy_j_est": energy_j if valid_pairs else None,
            "avg_power_w": statistics.fmean(powers) if powers else None,
            "max_power_w": max(powers) if powers else None,
            "max_temp_c": max(temps) if temps else None,
            "avg_gpu_util_pct": statistics.fmean(utils) if utils else None,
        }

    def current_util(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.available:
            return None, None
        try:
            r = self._read()
            return r.get("gpu_util_pct"), r.get("power_w")
        except Exception:
            return None, None


def wait_for_quiet(nvml: NVMLSampler, max_seconds: float = 20.0, cpu_thresh: float = 25.0, gpu_thresh: float = 10.0) -> Dict[str, Any]:
    start = time.perf_counter()
    passed = False
    last_cpu = None
    last_gpu = None
    consecutive = 0
    while time.perf_counter() - start < max_seconds:
        last_cpu = psutil.cpu_percent(interval=0.25)
        gpu, _ = nvml.current_util()
        last_gpu = gpu
        gpu_ok = gpu is None or gpu <= gpu_thresh
        if last_cpu <= cpu_thresh and gpu_ok:
            consecutive += 1
            if consecutive >= 4:
                passed = True
                break
        else:
            consecutive = 0
    return {
        "quiet_passed": passed,
        "quiet_wait_s": time.perf_counter() - start,
        "cpu_util_pct_before": last_cpu,
        "gpu_util_pct_before": last_gpu,
        "cpu_threshold_pct": cpu_thresh,
        "gpu_threshold_pct": gpu_thresh,
    }


def configure_torch() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        out["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        out["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        out["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    except Exception as e:
        out["error"] = repr(e)
    return out


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def model_revision(model_id: str) -> Optional[str]:
    if model_info is None:
        return None
    try:
        return model_info(model_id).sha
    except Exception:
        return None


def load_model(model_id: str, attn_impl: str = "sdpa", revision: Optional[str] = None):
    dtype = choose_dtype()
    is_phi = model_id.lower().startswith("microsoft/phi-4")

    # Phi-4-mini is natively represented as Phi3ForCausalLM in modern Transformers.
    # The repository's remote modeling_phi3.py currently imports LossKwargs from an
    # API location that is incompatible with Transformers 5.x.  Therefore:
    #   * Transformers >=5: prefer the integrated/native implementation and DO NOT
    #     execute remote modeling code.
    #   * Transformers 4.49+ already contains the Phi3 architecture used by Phi-4-mini.
    #     We keep remote model code disabled in every validation environment so that
    #     repository-side API drift (including LossKwargs imports) cannot affect the run.
    # This decision is recorded in model metadata by the caller.
    major = int(str(transformers.__version__).split(".", 1)[0])
    trust_remote_code = False

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, revision=revision, trust_remote_code=trust_remote_code
    )
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token_id is not None:
            tok.pad_token = tok.unk_token
    kwargs = dict(
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        trust_remote_code=trust_remote_code,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, revision=revision, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, revision=revision, **kwargs)
    model.eval()
    return tok, model, dtype


def gpu_memory_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda": False}
    idx = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(idx)
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(idx),
        "allocated_gib": torch.cuda.memory_allocated(idx) / (1024 ** 3),
        "reserved_gib": torch.cuda.memory_reserved(idx) / (1024 ** 3),
        "free_gib": free_b / (1024 ** 3),
        "total_gib": total_b / (1024 ** 3),
    }


def print_gpu_memory(prefix: str) -> None:
    m = gpu_memory_snapshot()
    if not m.get("cuda"):
        return
    print(
        f"{prefix} VRAM allocated={m['allocated_gib']:.2f} GiB, "
        f"reserved={m['reserved_gib']:.2f} GiB, free={m['free_gib']:.2f}/{m['total_gib']:.2f} GiB",
        flush=True,
    )


def fixed_length_ids(tokenizer, text: str, target_len: int) -> torch.Tensor:
    base = tokenizer(text, add_special_tokens=True, return_tensors=None)["input_ids"]
    filler = tokenizer(" Therefore, continue the reasoning carefully and concisely.", add_special_tokens=False, return_tensors=None)["input_ids"]
    if not filler:
        filler = [tokenizer.eos_token_id or 0]
    ids = list(base)
    while len(ids) < target_len:
        take = min(len(filler), target_len - len(ids))
        ids.extend(filler[:take])
    ids = ids[:target_len]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def plain_ids(tokenizer, text: str, max_len: int = 2048) -> torch.Tensor:
    data = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_len, return_tensors="pt")
    return data["input_ids"].cpu()


def natural_ids(tokenizer, text: str, max_len: int = 2048) -> torch.Tensor:
    """Use the model's chat template when available; otherwise use plain tokenization."""
    try:
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if ids.shape[1] > max_len:
                ids = ids[:, -max_len:]
            return ids.cpu()
    except Exception:
        pass
    return plain_ids(tokenizer, text, max_len=max_len)


def make_uniforms(seed_key: str, n: int, steps: int, device: torch.device) -> torch.Tensor:
    seed_bytes = hashlib.blake2b(seed_key.encode("utf-8"), digest_size=16).digest()
    seed = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.Generator(np.random.Philox(seed))
    arr = rng.random((n, steps), dtype=np.float64).astype(np.float32)
    # Keep away from exact endpoints of a CDF.
    eps = np.finfo(np.float32).eps
    arr = np.clip(arr, eps, 1.0 - eps)
    return torch.from_numpy(arr).to(device=device)


def sample_from_uniform(
    logits: torch.Tensor,
    u: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    scores = logits.float() / float(temperature)
    if top_k and 0 < top_k < scores.shape[-1]:
        kth = torch.topk(scores, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        csum = torch.cumsum(sorted_probs, dim=-1)
        remove = (csum - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        sorted_probs = sorted_probs / denom
        cdf = torch.cumsum(sorted_probs, dim=-1)
        cdf[:, -1] = 1.0
        pos = torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous(), right=False).squeeze(-1)
        pos = torch.clamp(pos, max=sorted_idx.shape[-1] - 1)
        return sorted_idx.gather(1, pos.unsqueeze(-1)).squeeze(-1)
    cdf = torch.cumsum(probs, dim=-1)
    cdf[:, -1] = 1.0
    pos = torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous(), right=False).squeeze(-1)
    return torch.clamp(pos, max=probs.shape[-1] - 1)


def repeat_cache(cache: Any, repeats: int) -> Any:
    if hasattr(cache, "batch_repeat_interleave"):
        cache.batch_repeat_interleave(repeats)
        return cache
    # Legacy tuple cache fallback: tuple[layer] -> (K, V, ...)
    if isinstance(cache, (tuple, list)):
        out = []
        for layer in cache:
            if isinstance(layer, (tuple, list)):
                new_layer = []
                for x in layer:
                    if torch.is_tensor(x) and x.ndim >= 1:
                        new_layer.append(x.repeat_interleave(repeats, dim=0))
                    else:
                        new_layer.append(x)
                out.append(tuple(new_layer))
            else:
                out.append(layer)
        return tuple(out)
    raise TypeError(f"Unsupported cache type for repeat: {type(cache)!r}")


def forward_kwargs(model) -> Dict[str, Any]:
    # Modern causal-LM classes may expose a last-logit-only optimization.
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


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def run_generation_method(
    model,
    input_ids_cpu: torch.Tensor,
    uniforms: torch.Tensor,
    method: str,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 0,
    capture_first_logits: bool = False,
    collapsed_rng: bool = False,
) -> Dict[str, Any]:
    assert method in {"batched_duplicate", "shared_prefill"}
    device = next(model.parameters()).device
    n, steps = uniforms.shape
    ids1 = input_ids_cpu.to(device=device, non_blocking=False)
    fkwargs = forward_kwargs(model)

    if collapsed_rng:
        uniforms = uniforms[:1, :].repeat(n, 1)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start_wall = time.perf_counter()
    e0, e1, e2, e3 = _event(), _event(), _event(), _event()
    e0.record()

    with torch.inference_mode():
        if method == "batched_duplicate":
            batch_ids = ids1.repeat(n, 1)
            out = model(input_ids=batch_ids, use_cache=True, return_dict=True, **fkwargs)
            logits = out.logits[:, -1, :]
            cache = out.past_key_values
            e1.record()
            e2.record()
        else:
            out = model(input_ids=ids1, use_cache=True, return_dict=True, **fkwargs)
            logits = out.logits[:, -1, :].repeat(n, 1)
            cache = out.past_key_values
            e1.record()
            cache = repeat_cache(cache, n)
            e2.record()

        first_logits = logits.detach().float().cpu() if capture_first_logits else None
        generated = torch.empty((n, steps), dtype=torch.long, device=device)
        next_tok = None
        for t in range(steps):
            if t > 0:
                out = model(input_ids=next_tok.unsqueeze(-1), past_key_values=cache, use_cache=True, return_dict=True, **fkwargs)
                logits = out.logits[:, -1, :]
                cache = out.past_key_values
            next_tok = sample_from_uniform(logits, uniforms[:, t], temperature=temperature, top_p=top_p, top_k=top_k)
            generated[:, t] = next_tok
        e3.record()

    torch.cuda.synchronize()
    elapsed_wall_ms = (time.perf_counter() - start_wall) * 1000.0
    result = {
        "method": method,
        "wall_ms": elapsed_wall_ms,
        "gpu_total_ms": e0.elapsed_time(e3),
        "gpu_prefill_ms": e0.elapsed_time(e1),
        "gpu_fork_ms": e1.elapsed_time(e2),
        "gpu_decode_ms": e2.elapsed_time(e3),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "generated": generated.detach().cpu(),
        "first_logits": first_logits,
    }
    return result


def sequence_stats(tokens: torch.Tensor) -> Dict[str, Any]:
    seqs = [tuple(map(int, row.tolist())) for row in tokens]
    unique = len(set(seqs))
    n = len(seqs)
    equal_pairs = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            equal_pairs += int(seqs[i] == seqs[j])
    return {
        "n_trajectories": n,
        "unique_trajectories": unique,
        "unique_fraction": unique / n if n else None,
        "identical_pair_fraction": equal_pairs / total_pairs if total_pairs else None,
    }


def free_model(model=None, tokenizer=None) -> None:
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_prompt_corpus(n: int) -> Tuple[List[str], Dict[str, Any]]:
    bundled = Path(__file__).resolve().parent / "publication_prompts.json"
    if bundled.exists():
        try:
            obj = json.loads(bundled.read_text(encoding="utf-8"))
            prompts = [str(x) for x in obj.get("prompts", [])][:n]
            if len(prompts) >= min(n, 4):
                return prompts, {
                    "source": obj.get("source", "bundled_publication_prompts"),
                    "bundle_sha256": sha256_file(bundled),
                    "count": len(prompts),
                }
        except Exception:
            pass
    if load_dataset is not None:
        try:
            ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
            prompts = [str(x["question"]) for x in ds]
            return prompts, {
                "source": "openai/gsm8k",
                "split": f"test[:{n}]",
                "dataset_fingerprint": getattr(ds, "_fingerprint", None),
                "count": len(prompts),
            }
        except Exception as e:
            return FALLBACK_PROMPTS[:n], {
                "source": "embedded_fallback",
                "reason": repr(e),
                "count": min(n, len(FALLBACK_PROMPTS)),
            }
    return FALLBACK_PROMPTS[:n], {"source": "embedded_fallback", "reason": "datasets unavailable", "count": min(n, len(FALLBACK_PROMPTS))}


def paired_bootstrap_median_ratio(a: Sequence[float], b: Sequence[float], seed: int = MASTER_SEED, reps: int = 10000) -> Dict[str, Optional[float]]:
    # Ratio a/b, paired by replicate.
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    ok = np.isfinite(arr_a) & np.isfinite(arr_b) & (arr_b > 0)
    arr_a, arr_b = arr_a[ok], arr_b[ok]
    if len(arr_a) == 0:
        return {"median_ratio": None, "ci95_low": None, "ci95_high": None, "n": 0}
    ratios = arr_a / arr_b
    med = float(np.median(ratios))
    if len(ratios) == 1:
        return {"median_ratio": med, "ci95_low": med, "ci95_high": med, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), size=(reps, len(ratios)))
    boots = np.median(ratios[idx], axis=1)
    return {
        "median_ratio": med,
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "n": int(len(ratios)),
    }


def benchmark_one_config(
    model_id: str,
    model,
    tokenizer,
    input_ids_cpu: torch.Tensor,
    sample_count: int,
    max_new_tokens: int,
    repetitions: int,
    benchmark_id: str,
    nvml: NVMLSampler,
    temperature: float = 0.8,
    top_p: float = 1.0,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    uniforms = make_uniforms(f"{model_id}|{benchmark_id}", sample_count, max_new_tokens, device)

    # Warm both paths outside measurement.  Emit progress so a long GPU kernel
    # cannot be mistaken for a stalled process.
    for warm_i, method in enumerate(["batched_duplicate", "shared_prefill"], start=1):
        print(f"    [warmup {warm_i}/2] {benchmark_id} :: {method}", flush=True)
        try:
            _ = run_generation_method(
                model, input_ids_cpu, uniforms, method,
                temperature=temperature, top_p=top_p,
            )
            del _
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

    rows: List[Dict[str, Any]] = []
    for rep in range(repetitions):
        order_rng = random.Random(MASTER_SEED + rep + int(hashlib.sha256(benchmark_id.encode()).hexdigest()[:8], 16))
        methods = ["batched_duplicate", "shared_prefill"]
        order_rng.shuffle(methods)
        for method_index, method in enumerate(methods, start=1):
            print(
                f"    [rep {rep + 1}/{repetitions}, method {method_index}/2] "
                f"{benchmark_id} :: {method}",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            quiet = wait_for_quiet(nvml)
            if quiet.get("quiet_wait_s", 0) >= 3.0:
                print(
                    f"      quiet-window wait {quiet['quiet_wait_s']:.1f}s "
                    f"(passed={quiet['quiet_passed']}, cpu={quiet.get('cpu_util_pct_before')}, "
                    f"gpu={quiet.get('gpu_util_pct_before')})",
                    flush=True,
                )
            mon = NVMLSampler(0, 0.05)
            mon.start()
            try:
                r = run_generation_method(
                    model, input_ids_cpu, uniforms, method,
                    temperature=temperature, top_p=top_p,
                )
                energy = mon.stop()
                st = sequence_stats(r["generated"])
                rows.append({
                    "status": "ok",
                    "model_id": model_id,
                    "benchmark_id": benchmark_id,
                    "replicate": rep,
                    "method": method,
                    "prompt_tokens": int(input_ids_cpu.shape[1]),
                    "n_trajectories": sample_count,
                    "max_new_tokens": max_new_tokens,
                    "logical_output_tokens": sample_count * max_new_tokens,
                    "wall_ms": r["wall_ms"],
                    "gpu_total_ms": r["gpu_total_ms"],
                    "gpu_prefill_ms": r["gpu_prefill_ms"],
                    "gpu_fork_ms": r["gpu_fork_ms"],
                    "gpu_decode_ms": r["gpu_decode_ms"],
                    "tokens_per_s_wall": sample_count * max_new_tokens / (r["wall_ms"] / 1000.0),
                    "tokens_per_s_gpu": sample_count * max_new_tokens / (r["gpu_total_ms"] / 1000.0),
                    "peak_allocated_bytes": r["peak_allocated_bytes"],
                    "peak_reserved_bytes": r["peak_reserved_bytes"],
                    **quiet,
                    **energy,
                    **st,
                })
                print(
                    f"      done: wall={r['wall_ms']:.1f} ms, gpu={r['gpu_total_ms']:.1f} ms, "
                    f"peak={r['peak_allocated_bytes'] / (1024 ** 3):.2f} GiB",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as e:
                energy = mon.stop()
                rows.append({
                    "status": "oom",
                    "error": repr(e),
                    "model_id": model_id,
                    "benchmark_id": benchmark_id,
                    "replicate": rep,
                    "method": method,
                    "prompt_tokens": int(input_ids_cpu.shape[1]),
                    "n_trajectories": sample_count,
                    "max_new_tokens": max_new_tokens,
                    **quiet,
                    **energy,
                })
                torch.cuda.empty_cache()
            except Exception as e:
                energy = mon.stop()
                rows.append({
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "model_id": model_id,
                    "benchmark_id": benchmark_id,
                    "replicate": rep,
                    "method": method,
                    "prompt_tokens": int(input_ids_cpu.shape[1]),
                    "n_trajectories": sample_count,
                    "max_new_tokens": max_new_tokens,
                    **quiet,
                    **energy,
                })
    return rows


def semantic_check(
    model_id: str,
    model,
    tokenizer,
    prompts: Sequence[str],
    sample_count: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    device = next(model.parameters()).device
    for pi, prompt in enumerate(prompts):
        print(f"    [semantic {pi + 1}/{len(prompts)}] {model_id}", flush=True)
        ids = natural_ids(tokenizer, prompt, max_len=1024)
        key = f"semantic|{model_id}|{pi}|{sha256_text(prompt)}"
        u = make_uniforms(key, sample_count, max_new_tokens, device)
        try:
            base = run_generation_method(model, ids, u, "batched_duplicate", capture_first_logits=True)
            shared = run_generation_method(model, ids, u, "shared_prefill", capture_first_logits=True)
            neg = run_generation_method(model, ids, u, "shared_prefill", collapsed_rng=True)
            eq = base["generated"] == shared["generated"]
            trajectory_match = torch.all(eq, dim=1)
            mismatch_tokens = int((~eq).sum().item())
            first_mismatch = None
            if mismatch_tokens:
                loc = torch.nonzero(~eq, as_tuple=False)[0].tolist()
                first_mismatch = {"trajectory": int(loc[0]), "step": int(loc[1])}
            max_logit_diff = None
            if base["first_logits"] is not None and shared["first_logits"] is not None:
                max_logit_diff = float((base["first_logits"] - shared["first_logits"]).abs().max().item())
            rows.append({
                "status": "ok",
                "model_id": model_id,
                "prompt_index": pi,
                "prompt_sha256": sha256_text(prompt),
                "prompt_tokens": int(ids.shape[1]),
                "n_trajectories": sample_count,
                "max_new_tokens": max_new_tokens,
                "matched_trajectories": int(trajectory_match.sum().item()),
                "trajectory_match_rate": float(trajectory_match.float().mean().item()),
                "matched_tokens": int(eq.sum().item()),
                "total_tokens": int(eq.numel()),
                "token_match_rate": float(eq.float().mean().item()),
                "mismatch_tokens": mismatch_tokens,
                "first_mismatch": first_mismatch,
                "max_abs_first_step_logit_diff": max_logit_diff,
                "proper_rng_stats": sequence_stats(shared["generated"]),
                "collapsed_rng_stats": sequence_stats(neg["generated"]),
                "negative_control_changes_joint_process": bool(not torch.equal(shared["generated"], neg["generated"])),
            })
        except torch.cuda.OutOfMemoryError as e:
            rows.append({"status": "oom", "model_id": model_id, "prompt_index": pi, "error": repr(e)})
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append({"status": "error", "model_id": model_id, "prompt_index": pi, "error": repr(e), "traceback": traceback.format_exc()})
    return rows


def aggregate_transformer(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    ok = raw[raw["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["model_id", "benchmark_id", "prompt_tokens", "n_trajectories", "max_new_tokens"]
    out: List[Dict[str, Any]] = []
    for key, g in ok.groupby(group_cols, dropna=False):
        piv_wall = g.pivot_table(index="replicate", columns="method", values="wall_ms", aggfunc="first")
        piv_gpu = g.pivot_table(index="replicate", columns="method", values="gpu_total_ms", aggfunc="first")
        piv_energy = g.pivot_table(index="replicate", columns="method", values="energy_j_est", aggfunc="first") if "energy_j_est" in g.columns else pd.DataFrame()
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        for metric in ["wall_ms", "gpu_total_ms", "gpu_prefill_ms", "gpu_fork_ms", "gpu_decode_ms", "tokens_per_s_wall", "peak_allocated_bytes"]:
            if metric not in g.columns:
                continue
            for method in ["batched_duplicate", "shared_prefill"]:
                vals = pd.to_numeric(g.loc[g.method == method, metric], errors="coerce").dropna().to_numpy()
                row[f"{method}_{metric}_median"] = float(np.median(vals)) if len(vals) else None
                row[f"{method}_{metric}_iqr"] = float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)) if len(vals) >= 2 else 0.0 if len(vals) else None
        if {"batched_duplicate", "shared_prefill"}.issubset(piv_wall.columns):
            both = piv_wall[["batched_duplicate", "shared_prefill"]].dropna()
            ci = paired_bootstrap_median_ratio(both["batched_duplicate"].to_numpy(), both["shared_prefill"].to_numpy())
            row.update({f"wall_speedup_{k}": v for k, v in ci.items()})
        if {"batched_duplicate", "shared_prefill"}.issubset(piv_gpu.columns):
            both = piv_gpu[["batched_duplicate", "shared_prefill"]].dropna()
            ci = paired_bootstrap_median_ratio(both["batched_duplicate"].to_numpy(), both["shared_prefill"].to_numpy())
            row.update({f"gpu_speedup_{k}": v for k, v in ci.items()})
        if not piv_energy.empty and {"batched_duplicate", "shared_prefill"}.issubset(piv_energy.columns):
            both = piv_energy[["batched_duplicate", "shared_prefill"]].dropna()
            if not both.empty:
                ci = paired_bootstrap_median_ratio(both["batched_duplicate"].to_numpy(), both["shared_prefill"].to_numpy())
                row.update({f"energy_reduction_factor_{k}": v for k, v in ci.items()})
        out.append(row)
    return pd.DataFrame(out)


def run_transformer_study(config: ProfileConfig, out_dir: Path, run_semantic: bool = True) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nvml = NVMLSampler(0)
    prompt_pool, prompt_meta = load_prompt_corpus(max(config.real_prompts, config.semantic_prompts))
    write_json(out_dir / "prompt_source.json", prompt_meta)
    write_json(out_dir / "prompt_manifest.json", [
        {"index": i, "sha256": sha256_text(p), "text": p} for i, p in enumerate(prompt_pool)
    ])
    all_rows: List[Dict[str, Any]] = []
    semantic_rows: List[Dict[str, Any]] = []
    model_meta: List[Dict[str, Any]] = []

    for model_index, model_id in enumerate(config.models, start=1):
        print(f"\n[LLM] Model {model_index}/{len(config.models)}: {model_id}", flush=True)
        print_gpu_memory("[LLM] Before load:")
        print(f"[LLM] Resolving revision and loading {model_id} ...", flush=True)
        load_started = time.perf_counter()
        resolved_revision = model_revision(model_id)
        try:
            tokenizer, model, dtype = load_model(model_id, revision=resolved_revision)
        except Exception as e:
            model_meta.append({
                "model_id": model_id,
                "status": "load_error",
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            print(f"[LLM] SKIP {model_id}: {e}", flush=True)
            continue
        load_s = time.perf_counter() - load_started
        print(f"[LLM] Loaded {model_id} in {load_s:.1f}s", flush=True)
        print_gpu_memory("[LLM] After load:")
        revision = resolved_revision or getattr(getattr(model, "config", None), "_commit_hash", None)
        params = sum(int(p.numel()) for p in model.parameters())
        model_meta.append({
            "model_id": model_id,
            "status": "ok",
            "revision": revision,
            "parameter_count": params,
            "dtype": str(dtype),
            "attention_implementation_requested": "sdpa",
            "load_seconds": load_s,
            "config_model_type": getattr(model.config, "model_type", None),
            "architecture": getattr(model.config, "architectures", None),
            "vocab_size": getattr(model.config, "vocab_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
        })
        write_json(out_dir / "model_metadata.json", model_meta)

        # The v2 batched-vs-shared semantic check is retained only for backward
        # compatibility. Publication v3 uses semantic_validation_v2.py instead,
        # which compares batch-1 serial reference against shared-prefill batch-1
        # trajectories and reports numerical distribution distances.
        if run_semantic:
            print(f"[LLM] Legacy semantic checks: {model_id}", flush=True)
            srows = semantic_check(
                model_id, model, tokenizer,
                prompt_pool[: config.semantic_prompts],
                sample_count=min(8, max(config.model_sample_counts.get(model_id, [8]))),
                max_new_tokens=config.semantic_max_new_tokens,
            )
            semantic_rows.extend(srows)
            pd.json_normalize(semantic_rows).to_csv(out_dir / "semantic_checks.csv", index=False)

        # Controlled prompt-length microbenchmarks.
        for plen in config.micro_prompt_lengths:
            ids = fixed_length_ids(tokenizer, prompt_pool[0], plen)
            for decode_len in config.micro_decode_lengths:
                for n in config.model_sample_counts.get(model_id, [1, 4, 8]):
                    bid = f"micro_len{plen}_out{decode_len}_n{n}"
                    print(f"[LLM] {model_id} {bid}", flush=True)
                    try:
                        rows = benchmark_one_config(
                            model_id, model, tokenizer, ids, n,
                            decode_len, config.micro_repetitions,
                            bid, nvml,
                        )
                        all_rows.extend(rows)
                    except torch.cuda.OutOfMemoryError as e:
                        all_rows.append({
                            "status": "oom",
                            "model_id": model_id,
                            "benchmark_id": bid,
                            "prompt_tokens": plen,
                            "n_trajectories": n,
                            "max_new_tokens": decode_len,
                            "error": repr(e),
                        })
                        torch.cuda.empty_cache()
                    pd.DataFrame(all_rows).to_csv(out_dir / "transformer_raw.csv", index=False)

        # Real-prompt workload. One config per natural prompt, same sample count.
        for pi, prompt in enumerate(prompt_pool[: config.real_prompts]):
            ids = natural_ids(tokenizer, prompt, max_len=1024)
            n = min(config.real_sample_count, max(config.model_sample_counts.get(model_id, [config.real_sample_count])))
            bid = f"gsm8k_prompt{pi}_n{n}"
            print(f"[LLM] {model_id} {bid}", flush=True)
            try:
                rows = benchmark_one_config(
                    model_id, model, tokenizer, ids, n,
                    config.real_max_new_tokens, config.real_repetitions,
                    bid, nvml,
                )
                for row in rows:
                    row["prompt_sha256"] = sha256_text(prompt)
                    row["workload_source"] = prompt_meta.get("source")
                all_rows.extend(rows)
            except torch.cuda.OutOfMemoryError as e:
                all_rows.append({
                    "status": "oom",
                    "model_id": model_id,
                    "benchmark_id": bid,
                    "prompt_tokens": int(ids.shape[1]),
                    "n_trajectories": n,
                    "max_new_tokens": config.real_max_new_tokens,
                    "error": repr(e),
                })
                torch.cuda.empty_cache()
            pd.DataFrame(all_rows).to_csv(out_dir / "transformer_raw.csv", index=False)

        # IMPORTANT: delete the strong references in this caller.  The v1 helper
        # only deleted its local parameter names, which left the just-finished
        # model resident in VRAM while the next model was being loaded.
        print(f"[LLM] Completed {model_id}; releasing model before next load ...", flush=True)
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print_gpu_memory("[LLM] After release:")
        print(f"[LLM] Model complete: {model_id}", flush=True)

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(out_dir / "transformer_raw.csv", index=False)
    sem_df = pd.json_normalize(semantic_rows)
    sem_df.to_csv(out_dir / "semantic_checks.csv", index=False)
    agg = aggregate_transformer(raw_df)
    agg.to_csv(out_dir / "transformer_aggregate.csv", index=False)
    write_json(out_dir / "model_metadata.json", model_meta)

    summary = {
        "models_requested": config.models,
        "models_loaded": [m["model_id"] for m in model_meta if m.get("status") == "ok"],
        "raw_rows": int(len(raw_df)),
        "semantic_rows": int(len(sem_df)),
        "aggregate_rows": int(len(agg)),
        "semantic_total_tokens": int(pd.to_numeric(sem_df.get("total_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sem_df.empty else 0,
        "semantic_mismatch_tokens": int(pd.to_numeric(sem_df.get("mismatch_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sem_df.empty else 0,
    }
    write_json(out_dir / "transformer_summary.json", summary)
    return summary


# ---------- legacy CAM v2 rerun ----------

def extract_legacy(root: Path) -> Path:
    zpath = root / "CAM_v2_Reproducible_original.zip"
    dest = root / "_legacy_work"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(dest)
    return dest / "CAM_v2"


def run_subprocess_to_log(cmd: Sequence[str], cwd: Path, log_path: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True, check=False)
    return {"cmd": list(cmd), "returncode": p.returncode, "elapsed_s": time.perf_counter() - started, "log": str(log_path)}


def run_legacy_local_timing(legacy_root: Path, out_dir: Path) -> Dict[str, Any]:
    # Import the original kernels from the extracted immutable bundle, but write new measurements elsewhere.
    sys.path.insert(0, str(legacy_root))
    try:
        from kernel import NeuralBackend, independent, exact_prefix, histogram, decision
        from optimized_kernel import factored_independent, optimized_exact
        from threadpoolctl import threadpool_limits, threadpool_info
    finally:
        pass

    z = np.load(legacy_root / "dataset_and_splits.npz", allow_pickle=False)
    rows: List[List[Any]] = []
    methods = {
        "independent": independent,
        "exact_prefix": exact_prefix,
        "factored_independent": factored_independent,
        "optimized_exact": optimized_exact,
    }
    env = {
        "timestamp_utc": now_utc(),
        "status": "local user-owned CPU benchmark; no cloud/shared-tenant CPU",
        "process": set_process_priority(),
        "cpu_percent_pre": psutil.cpu_percent(interval=1.0),
        "threadpool_info_pre": None,
    }
    with threadpool_limits(limits=1):
        env["threadpool_info_pre"] = threadpool_info()
        for seed in [17, 29, 41]:
            model = NeuralBackend.load(legacy_root / "weights" / f"model_{seed}.npz")
            names = list(methods)
            for n in [1, 32, 128]:
                # warmup
                for name in names:
                    fn = methods[name]
                    for i in z["test"][:5]:
                        r = fn(model, z["x"][i], np.random.default_rng(int(i)).random((n, 4)))
                        paths = r[0] if name in ["factored_independent", "optimized_exact"] else r
                        decision(histogram(paths))
                for rep in range(7):
                    order = np.random.default_rng(744 + seed + n + rep).permutation(names)
                    for name in order:
                        # Quietness measurement is informative; do not silently discard noisy runs.
                        cpu_pre = psutil.cpu_percent(interval=0.2)
                        rng = np.random.default_rng(481 + seed + n + rep)
                        wall0 = time.perf_counter_ns()
                        cpu0 = time.process_time_ns()
                        fn = methods[name]
                        for i in z["test"][:128]:
                            r = fn(model, z["x"][i], rng.random((n, 4)))
                            paths = r[0] if name in ["factored_independent", "optimized_exact"] else r
                            decision(histogram(paths))
                        cpu_ms = (time.process_time_ns() - cpu0) / 1e6
                        wall_ms = (time.perf_counter_ns() - wall0) / 1e6
                        rows.append([seed, n, rep, name, wall_ms, cpu_ms, cpu_pre])
            print(f"[LEGACY] local timing seed {seed} done", flush=True)
    env["cpu_percent_post"] = psutil.cpu_percent(interval=1.0)
    env["threadpool_info_post"] = threadpool_info()
    df = pd.DataFrame(rows, columns=["seed", "n", "replicate", "method", "wall_ms", "process_cpu_ms", "cpu_util_pre_pct"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "legacy_local_timings.csv", index=False)
    write_json(out_dir / "legacy_local_timing_environment.json", env)
    speed_rows = []
    for metric in ["wall_ms", "process_cpu_ms"]:
        for (seed, n), g in df.groupby(["seed", "n"]):
            p = g.pivot(index="replicate", columns="method", values=metric)
            for m in ["exact_prefix", "factored_independent", "optimized_exact"]:
                if m in p and "independent" in p:
                    a = p["independent"].to_numpy()
                    b = p[m].to_numpy()
                    ci = paired_bootstrap_median_ratio(a, b, seed=MASTER_SEED + seed + n)
                    speed_rows.append({"metric": metric, "seed": seed, "n": n, "method": m, **ci})
    sp = pd.DataFrame(speed_rows)
    sp.to_csv(out_dir / "legacy_local_speedups.csv", index=False)
    return {"timing_rows": len(df), "speedup_rows": len(sp)}


def run_legacy_study(root: Path, out_dir: Path, run_full: bool) -> Dict[str, Any]:
    legacy_root = extract_legacy(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "legacy_bundle_sha256": sha256_file(root / "CAM_v2_Reproducible_original.zip"),
        "legacy_root": str(legacy_root),
        "run_full_original_protocol": run_full,
        "steps": [],
    }
    if run_full:
        scripts = [
            "run_neural.py",
            "check_and_summarize.py",
            "strong_controls.py",
            "benchmark_optimized.py",
            "test_generic_engine.py",
        ]
        for script in scripts:
            print(f"[LEGACY] running {script}", flush=True)
            res = run_subprocess_to_log([sys.executable, script], legacy_root, out_dir / f"{script}.log")
            manifest["steps"].append(res)
            if res["returncode"] != 0:
                manifest["warning"] = f"Original protocol step {script} returned non-zero; local timing will still be attempted."
                break
    try:
        manifest["local_timing"] = run_legacy_local_timing(legacy_root, out_dir)
    except Exception as e:
        manifest["local_timing"] = {"status": "error", "error": repr(e), "traceback": traceback.format_exc()}
    # Preserve selected rerun artifacts without modifying the original uploaded ZIP.
    for rel in [
        "results/summary.json",
        "analysis/quality_and_compute.csv",
        "analysis/optimized_speedups.csv",
        "analysis/optimized_checks.csv",
    ]:
        src = legacy_root / rel
        if src.exists():
            dst = out_dir / (rel.replace("/", "__"))
            shutil.copy2(src, dst)
    write_json(out_dir / "legacy_manifest.json", manifest)
    return manifest


def render_summary(root_out: Path) -> None:
    lines: List[str] = [
        "# Stochastic Provenance local study summary",
        "",
        f"Generated: {now_utc()}",
        "",
        "This summary is generated from the raw local measurements. It is not a claim of state-of-the-art performance.",
        "",
    ]
    t_agg = root_out / "transformer" / "transformer_aggregate.csv"
    sem = root_out / "transformer" / "semantic_checks.csv"
    if t_agg.exists():
        df = pd.read_csv(t_agg)
        lines += ["## Transformer performance", ""]
        if df.empty:
            lines.append("No successful Transformer benchmark rows were available.")
        else:
            cols = [c for c in ["model_id", "benchmark_id", "prompt_tokens", "n_trajectories", "wall_speedup_median_ratio", "wall_speedup_ci95_low", "wall_speedup_ci95_high", "gpu_speedup_median_ratio"] if c in df.columns]
            lines.append(df[cols].to_markdown(index=False))
        lines.append("")
    if sem.exists():
        df = pd.read_csv(sem)
        lines += ["## Coupled semantic checks", ""]
        if not df.empty and "mismatch_tokens" in df.columns:
            ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
            total = pd.to_numeric(ok.get("total_tokens", 0), errors="coerce").fillna(0).sum()
            mism = pd.to_numeric(ok.get("mismatch_tokens", 0), errors="coerce").fillna(0).sum()
            lines.append(f"Observed token mismatches between duplicated-prefix baseline and shared-prefix execution: **{int(mism)} / {int(total)}**.")
            if "max_abs_first_step_logit_diff" in ok.columns:
                vals = pd.to_numeric(ok["max_abs_first_step_logit_diff"], errors="coerce").dropna()
                if len(vals):
                    lines.append(f"Maximum observed absolute first-step logit difference: **{vals.max():.6g}**.")
            if "collapsed_rng_stats.unique_fraction" in ok.columns and "proper_rng_stats.unique_fraction" in ok.columns:
                a = pd.to_numeric(ok["proper_rng_stats.unique_fraction"], errors="coerce").dropna()
                b = pd.to_numeric(ok["collapsed_rng_stats.unique_fraction"], errors="coerce").dropna()
                if len(a) and len(b):
                    lines.append(f"Mean unique-trajectory fraction with path-local random events: **{a.mean():.3f}**; with the intentionally collapsed-RNG negative control: **{b.mean():.3f}**.")
        lines.append("")
    lsp = root_out / "legacy" / "legacy_local_speedups.csv"
    if lsp.exists():
        df = pd.read_csv(lsp)
        lines += ["## Original CAM v2 kernel on the local CPU", ""]
        sub = df[(df["metric"] == "wall_ms") & (df["n"].isin([32, 128]))] if not df.empty else df
        if not sub.empty:
            lines.append(sub[["seed", "n", "method", "median_ratio", "ci95_low", "ci95_high"]].to_markdown(index=False))
        lines.append("")
    lines += [
        "## Interpretation rule",
        "",
        "Use the raw CSV/JSON files when updating the manuscript. Do not substitute theoretical operation savings for measured latency or energy, and do not promote zero observed mismatches to a hardware-independent bitwise guarantee.",
        "",
    ]
    (root_out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def hash_tree(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith(".zip"):
            rows.append({"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return rows


def make_result_zip(root: Path, out_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zpath = root / f"stochastic_provenance_results_{stamp}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(out_dir)).replace("\\", "/"))
    return zpath


def main() -> int:
    ap = argparse.ArgumentParser(description="Local dedicated-machine study for Stochastic Provenance")
    ap.add_argument("--profile", choices=["quick", "standard", "publication"], default="publication")
    ap.add_argument("--output-root", default="results")
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-semantic", action="store_true", help="Skip legacy v2 semantic check; use semantic_validation_v2.py for publication semantics.")
    ap.add_argument(
        "--only-model",
        action="append",
        default=[],
        help="Run only this model ID (repeat the flag for more than one model). Useful for recovery/resume runs.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_root = Path(args.output_root).resolve()
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)

    cfg = get_profile(args.profile)
    if args.only_model:
        unknown = [m for m in args.only_model if m not in cfg.model_sample_counts]
        if unknown:
            raise ValueError(f"Unknown model(s) for profile {args.profile}: {unknown}")
        cfg.models = list(args.only_model)
    write_json(out / "study_config.json", {"profile": args.profile, **asdict(cfg), "master_seed": MASTER_SEED, "study_version": STUDY_VERSION})
    source_files = [
        "stochastic_provenance_study_v3.py",
        "semantic_validation_v2.py",
        "reaggregate_existing_results.py",
        "publication_prompts.json",
        "requirements_qwen_semantic.txt",
        "requirements_phi_compat_4_49.txt",
        "01_RUN_QWEN_SEMANTIC_VALIDATION.bat",
        "02_RUN_PHI_PUBLICATION_FIXED.bat",
        "02B_RUN_PHI_COMPAT_4_49.bat",
    ]
    write_json(out / "source_manifest.json", [
        {"path": name, "sha256": sha256_file(root / name), "bytes": (root / name).stat().st_size}
        for name in source_files if (root / name).exists()
    ])
    write_json(out / "hardware.json", collect_hardware())
    write_json(out / "software.json", collect_software())
    write_json(out / "process_policy.json", set_process_priority())
    write_json(out / "torch_policy.json", configure_torch())

    if not torch.cuda.is_available() and not args.skip_llm:
        raise RuntimeError("CUDA is required for the Transformer study.")

    top_summary: Dict[str, Any] = {
        "run_id": run_id, "profile": args.profile, "started_utc": now_utc(),
        "models_selected": cfg.models,
    }
    exit_code = 0
    try:
        if not args.skip_legacy:
            print("\n=== Re-running original CAM v2 on local CPU ===", flush=True)
            top_summary["legacy"] = run_legacy_study(root, out / "legacy", run_full=cfg.run_legacy_full)
        if not args.skip_llm:
            print("\n=== Running real Transformer/LLM study on local GPU ===", flush=True)
            top_summary["transformer"] = run_transformer_study(cfg, out / "transformer", run_semantic=not args.skip_semantic)
        top_summary["status"] = "completed"
        if "transformer" in top_summary:
            req = set(top_summary["transformer"].get("models_requested", []))
            loaded = set(top_summary["transformer"].get("models_loaded", []))
            if req != loaded:
                top_summary["status"] = "completed_with_errors"
                exit_code = 2
    except KeyboardInterrupt:
        exit_code = 130
        top_summary["status"] = "interrupted"
        top_summary["error"] = "KeyboardInterrupt"
        print("\n[RECOVERY] Interrupted by user; packaging all results written so far ...", flush=True)
    except Exception as e:
        exit_code = 1
        top_summary["status"] = "error"
        top_summary["error"] = repr(e)
        top_summary["traceback"] = traceback.format_exc()
        print(f"\n[RECOVERY] Study error: {e}", flush=True)
        traceback.print_exc()
        print("[RECOVERY] Packaging all results written so far ...", flush=True)
    finally:
        try:
            render_summary(out)
        except Exception as e:
            top_summary["summary_render_error"] = repr(e)
        top_summary["completed_utc"] = now_utc()
        write_json(out / "run_summary.json", top_summary)
        write_json(out / "file_manifest.json", hash_tree(out))
        zpath = make_result_zip(out_root, out)
        label = "RESULT_ZIP" if exit_code == 0 else "PARTIAL_RESULT_ZIP"
        print(f"\n{label}={zpath}", flush=True)
        print(f"RESULT_DIR={out}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
