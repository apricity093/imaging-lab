"""Training entry point matching the released 2048 pMF configuration."""
from __future__ import annotations

import argparse
import math
import os
import json
import random
import time
import numpy as np
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import CTSliceDataset


def _rank():
    return distributed.get_rank() if distributed.is_initialized() else 0


def _print(*values):
    if _rank() == 0:
        print(*values, flush=True)


def _adapt_pretrained(model, path):
    source = torch.load(path, map_location="cpu", weights_only=True)
    target = model.state_dict()
    loaded, skipped = 0, []
    for name, value in source.items():
        if name not in target:
            continue
        if value.shape == target[name].shape:
            target[name] = value
            loaded += 1
            continue
        if "x_embedder.proj1.weight" in name and value.shape[1] == 3 and target[name].shape[1] == 1:
            value = value.mean(dim=1, keepdim=True)
            if value.shape[-1] != target[name].shape[-1]:
                old, new = value.shape[-1], target[name].shape[-1]
                value = functional.interpolate(
                    value.reshape(-1, 1, old, old).float(), (new, new),
                    mode="bilinear", align_corners=False,
                ).reshape(target[name].shape) / ((new / old) ** 2)
            target[name] = value.to(target[name].dtype)
            loaded += 1
        else:
            skipped.append(name)
    model.load_state_dict(target)
    _print(f"pretrained tensors loaded={loaded}, skipped={len(skipped)}: {skipped}")


@torch.no_grad()
def _update_ema(ema, model, decay):
    values = dict(ema.named_parameters())
    for name, parameter in model.named_parameters():
        values[name].mul_(decay).add_(parameter, alpha=1 - decay)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", required=True)
    result.add_argument("--workdir", required=True)
    result.add_argument("--pretrained")
    result.add_argument("--from-scratch", action="store_true")
    result.add_argument("--model-config", help="JSON pixelMeanFlow constructor overrides")
    result.add_argument("--device", default="cuda")
    result.add_argument("--resume")
    result.add_argument("--epochs", type=int, default=500)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--lr", type=float, default=4e-4)
    result.add_argument("--warmup-epochs", type=int, default=10)
    result.add_argument("--ema-decay", type=float, default=0.9999)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--checkpoint-every", type=int, default=50)
    result.add_argument("--augment", action="store_true")
    return result


def entrypoint(argv=None):
    args = parser().parse_args(argv)
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training requires CUDA for --device cuda")
        torch.cuda.set_device(device)
    if world > 1 and device.type != "cuda":
        raise ValueError("distributed training requires CUDA/NCCL")
    if not (args.resume or args.pretrained or args.from_scratch):
        raise ValueError("provide --pretrained, --resume, or explicit --from-scratch")
    if args.epochs <= 0 or args.batch_size <= 0 or args.checkpoint_every <= 0:
        raise ValueError("epochs, batch-size and checkpoint-every must be positive")
    random.seed(args.seed + _rank())
    np.random.seed(args.seed + _rank())
    if world > 1:
        distributed.init_process_group("nccl")
    torch.manual_seed(args.seed + _rank())
    dataset = CTSliceDataset(args.data_root, "train", args.augment)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world > 1 else None
    if args.batch_size % world:
        raise ValueError("global batch size must be divisible by WORLD_SIZE")
    loader = DataLoader(
        dataset, batch_size=args.batch_size // world, sampler=sampler,
        shuffle=sampler is None, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    defaults = Path(__file__).resolve().parents[1] / "configs/training_2048.json"
    model_args = json.loads(defaults.read_text())
    if args.model_config:
        model_args.update(json.loads(Path(args.model_config).read_text()))
    from ....solvers.deep.models.pmf import pixelMeanFlow
    model = pixelMeanFlow(**model_args).to(device)
    if args.pretrained and not args.resume:
        _adapt_pretrained(model, args.pretrained)
    ema = deepcopy(model).requires_grad_(False)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr, betas=(0.9, 0.95))
    steps_per_epoch = len(loader)
    if not steps_per_epoch:
        raise ValueError("training corpus is smaller than one global batch")
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else
        0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))),
    )
    start_epoch = step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("model_args", model_args) != model_args:
            raise ValueError("resume model configuration differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
    workdir = Path(args.workdir)
    if _rank() == 0:
        workdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for images, labels in loader:
            loss, parts = wrapped(images.to(device, non_blocking=True), labels.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            _update_ema(ema, model, args.ema_decay)
            step += 1
            if step % 10 == 0:
                _print(f"epoch={epoch} step={step} loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.3g}")
        last_epoch = epoch + 1 == args.epochs
        if _rank() == 0 and ((epoch + 1) % args.checkpoint_every == 0 or last_epoch):
            torch.save({
                "epoch": epoch, "step": step, "model": model.state_dict(),
                "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(), "args": vars(args),
                "model_args": model_args,
                "model_config": {**model_args, "model": model_args["model_str"],
                                 "noise_scale": model.noise_scale},
            }, workdir / f"ckpt_epoch_{epoch}.pt")
    if _rank() == 0:
        (workdir / "training.json").write_text(json.dumps(dict(
            start_epoch=start_epoch, completed_epochs=args.epochs, step=step,
            last_loss=float(loss.detach()) if step and start_epoch < args.epochs else None,
            device=str(device), seconds=time.perf_counter() - started,
            model=model_args, arguments=vars(args)), indent=2))
    if distributed.is_initialized():
        distributed.destroy_process_group()


if __name__ == "__main__":
    entrypoint()
