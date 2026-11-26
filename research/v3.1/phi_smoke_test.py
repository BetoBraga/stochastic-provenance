from __future__ import annotations
import gc, json, sys
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL='microsoft/Phi-4-mini-instruct'
print('Python', sys.version)
print('Torch', torch.__version__, 'CUDA runtime', torch.version.cuda, 'CUDA available', torch.cuda.is_available())
print('Transformers', transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU is required')
trust_remote_code = False
print('Phi loader mode: native integrated Phi3 implementation (remote code disabled)')
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=trust_remote_code, use_fast=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token=tok.eos_token
dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
kw=dict(device_map={'':0}, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code, attn_implementation='sdpa')
try:
    model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, **kw)
except TypeError:
    model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, **kw)
model.eval()
ids=tok('A short smoke test: 2 + 2 =', return_tensors='pt')['input_ids'].to(model.device)
with torch.inference_mode():
    out=model(input_ids=ids, use_cache=True, return_dict=True)
shape=list(out.logits.shape)
print('PHI_SMOKE_OK logits_shape=', shape, 'dtype=', str(dtype), 'device=', torch.cuda.get_device_name(0))
del out, ids, model, tok
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
print('VRAM after release GiB=', round(torch.cuda.memory_allocated()/1024**3,3))
