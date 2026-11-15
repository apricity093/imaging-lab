"""Safe release-weight export and loading."""
from __future__ import annotations

import json
from pathlib import Path

import torch


from ...solvers.deep.models.weights import DEFAULT_MODEL_CONFIG, load_pmf


def export_ema(checkpoint, output, metadata_output=None):
    """Extract only EMA tensors; optimizer state and arbitrary pickle stay private."""
    from safetensors.torch import save_file

    checkpoint, output = Path(checkpoint), Path(output)
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = value.get("ema")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no EMA state dictionary")
    tensors = {
        name: tensor.detach().contiguous() for name, tensor in state.items()
        if not name.startswith(("net.rope_freqs", "lpips_model.", "net.v_heads.", "net.v_final_layer."))
        and not tensor.is_complex()
    }
    args = value.get("model_config", value.get("args", {}))
    if not isinstance(args, dict):
        args = vars(args)
    config = {**DEFAULT_MODEL_CONFIG, **{
        key: args[key] for key in DEFAULT_MODEL_CONFIG if key in args
    }}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={
        "format": "pt", "source_epoch": str(value.get("epoch", "unknown")),
        "architecture": config["model"],
    })
    metadata = {
        "format": "safetensors", "state": "ema", "epoch": value.get("epoch"),
        "step": value.get("step"), "model": config,
        "excluded": [
            "lpips_model.* (training loss network)",
            "net.v_heads.* and net.v_final_layer.* (training-only instantaneous-velocity head)",
            "net.rope_freqs (deterministically regenerated model buffer)",
        ],
    }
    destination = Path(metadata_output) if metadata_output else output.with_suffix(".json")
    destination.write_text(json.dumps(metadata, indent=2))
    return output, destination
