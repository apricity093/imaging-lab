"""Load released pMF EMA weights and adjacent model configuration."""
from __future__ import annotations

import json
from pathlib import Path

import torch


DEFAULT_MODEL_CONFIG = {
    "model": "pmfDiT_B_128", "img_size": 2048, "img_channels": 1,
    "num_classes": 1, "P_mean": -0.4, "P_std": 1.0, "cfg_max": 0.0,
    "data_proportion": 0.5, "cfg_beta": 1.0, "norm_p": 1.0,
    "norm_eps": 0.01, "tr_uniform": False, "noise_scale": None,
}


def load_pmf(weights, config_path=None, device="cuda"):
    from safetensors.torch import load_file
    from .pmf import pixelMeanFlow

    weights = Path(weights)
    metadata_path = Path(config_path) if config_path else weights.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    config = {**DEFAULT_MODEL_CONFIG, **metadata.get("model", {})}
    model = pixelMeanFlow(
        config["model"], img_size=config["img_size"],
        img_channels=config["img_channels"], num_classes=config["num_classes"],
        eval=True, P_mean=config["P_mean"], P_std=config["P_std"],
        cfg_max=config["cfg_max"], data_proportion=config["data_proportion"],
        cfg_beta=config["cfg_beta"], class_dropout_prob=0,
        norm_p=config["norm_p"], norm_eps=config["norm_eps"],
        tr_uniform=config["tr_uniform"], noise_scale=config["noise_scale"],
        msssim_lambda=0, lpips_lambda=0,
    ).to(device)
    missing, unexpected = model.load_state_dict(
        load_file(str(weights), device=str(device)), strict=False
    )
    if unexpected or set(missing) != {"net.rope_freqs"}:
        raise ValueError(
            f"release weight mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model
