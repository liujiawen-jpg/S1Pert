"""Audited external DECIDER-4B loader used only for frozen A4 representations.

The external source and weights must be explicitly supplied by configuration.
Historical data preparation, LoRA, calibration and training CLIs are excluded.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    path = Path(path).resolve(); s = path.stat()
    return {"path": str(path), "bytes": s.st_size, "sha256": sha(path)}


def checkpoint_target(source, target_names, ignored_patterns):
    """Match text weights; only skip auxiliary keys declared by HF's model."""
    import re
    prefix='model.language_model.'
    target='model.'+source[len(prefix):] if source.startswith(prefix) else source
    if target in target_names:
        return target
    if any(re.match(pattern,source) for pattern in ignored_patterns):
        return None
    raise ValueError(f'unexpected model tensor {source}->{target}')


def checkpoint_dtype_allowed(source_dtype, destination_dtype):
    # from_pretrained(dtype=bf16) likewise casts floating source tensors to the
    # requested precision. Qwen stores A_log and some norms in float32.
    return source_dtype==destination_dtype or (source_dtype,destination_dtype)==('F32','BF16')


def _external_decider_components(source):
    """Import only the user-supplied snapshot code, never an installed fallback."""
    source = Path(source).expanduser().resolve()
    package = source / "decider"
    for name in ("__init__.py", "model.py", "prompt.py"):
        if not (package / name).is_file():
            raise FileNotFoundError(f"external DECIDER source is missing {package / name}")
    for name, module in list(sys.modules.items()):
        if name == "decider" or name.startswith("decider."):
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(package):
                raise ImportError("a different DECIDER source is already imported; use a fresh process")
    sys.path.insert(0, str(source))
    try:
        model_module = importlib.import_module("decider.model")
        prompt_module = importlib.import_module("decider.prompt")
        for module in (model_module, prompt_module):
            if not Path(module.__file__).resolve().is_relative_to(package):
                raise ImportError("DECIDER was not imported from the configured external source")
        return model_module.DecisionModel, prompt_module.letter_ids
    finally:
        sys.path.remove(str(source))


def _load_model(cfg, arm, audit_path):
    if arm != "B6":
        raise ValueError("the A4 loader supports only the frozen DECIDER arm")
    DecisionModel, letter_ids = _external_decider_components(cfg["decider_source"])
    import torch
    from torch import nn
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    snapshot=Path(cfg["decider_snapshot"])
    tok=AutoTokenizer.from_pretrained(cfg["tokenizer_snapshot"],local_files_only=True)
    hf=AutoConfig.from_pretrained(snapshot,local_files_only=True)
    with torch.device("meta"): lm=AutoModelForCausalLM.from_config(hf,dtype=torch.bfloat16)
    lm.to_empty(device="cuda"); lm.tie_weights()
    rotary_type=type(lm.model.rotary_emb); lm.model.rotary_emb=rotary_type(lm.config,device=torch.device("cuda"))
    targets=dict(lm.named_parameters()); loaded=set(); files=sorted(snapshot.glob("model*.safetensors"))
    ignored_patterns=sorted(getattr(lm,'_keys_to_ignore_on_load_unexpected',None) or [])
    skipped=[]; dtype_conversions=[]
    if not files: raise FileNotFoundError("model safetensors missing: "+str(snapshot))
    with torch.no_grad():
        for path in files:
            with safe_open(str(path),framework="pt",device="cuda") as handle:
                for source in handle.keys():
                    target=checkpoint_target(source,targets,ignored_patterns)
                    if target is None:
                        skipped.append(source)
                        continue
                    value=handle.get_tensor(source); dest=targets[target]
                    if value.shape!=dest.shape: raise ValueError("model tensor shape mismatch: "+source)
                    if value.dtype!=dest.dtype:
                        if value.dtype!=torch.float32 or dest.dtype!=torch.bfloat16:
                            raise ValueError(f'Unsupported model tensor dtype: {source}: {value.dtype}->{dest.dtype}')
                        dtype_conversions.append({'tensor':source,'source':'F32','destination':'BF16'})
                        value=value.to(dtype=dest.dtype)
                    dest.copy_(value); loaded.add(target)
    missing=set(targets)-loaded
    if missing: raise ValueError("uninitialized model tensors: "+str(sorted(missing)[:8]))
    model=DecisionModel.__new__(DecisionModel); nn.Module.__init__(model); model.tok=tok; model.lm=lm
    model.register_buffer("letters",torch.tensor(letter_ids(tok),device="cuda"),persistent=False)
    model.requires_grad_(False); model.eval(); model.lm.config.use_cache=False
    model.runtime_cfg=cfg
    # Qwen3.5's audited Triton convolution avoids a slow/unstable generic
    # dispatch on the installed Blackwell runtime while preserving BF16 math.
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    def triton_conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        from fla.modules.conv.causal_conv1d import causal_conv1d
        import torch.nn.functional as F
        value,state=causal_conv1d(hidden_states.to(weight.dtype).transpose(1,2),weight,bias,
                                  activation=None,backend="triton",output_final_state=False)
        if state is not None: raise ValueError("unexpected convolution state")
        value=value.transpose(1,2)
        if activation is not None: value=F.silu(value)
        return value.to(hidden_states.dtype)
    qwen.causal_conv1d_fn=triton_conv
    dump(audit_path,{"snapshot":str(snapshot),"snapshot_files":[file_identity(x) for x in files],
                     "decider_source":str(Path(cfg["decider_source"]).resolve()),
                     "frozen_representation_only":True,"lora":False,
                     "parameter_count":sum(p.numel() for p in model.parameters()),"all_cuda":all(p.is_cuda for p in model.parameters()),
                     "model_class":type(lm).__name__,"loaded_parameter_tensors":len(loaded),
                     "missing_parameter_tensors":[],"official_ignored_key_patterns":ignored_patterns,
                     "skipped_auxiliary_tensors":skipped,
                     "dtype_conversions":dtype_conversions,
                     "tokenizer":file_identity(Path(cfg["tokenizer_snapshot"])/"tokenizer.json")})
    return model
