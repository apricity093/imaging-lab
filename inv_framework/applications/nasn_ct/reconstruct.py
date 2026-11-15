"""Run measured NaSn slices through the public CT experiment contract."""
import json
from pathlib import Path

import numpy as np
import tifffile

from .measurement import (IMAGE_SIZE, MEASURED_DETECTORS, PAD_LEFT,
                          PADDED_CENTER, TruncatedMeasurement)


def slice_experiment(weights, observation, scale, *, model_config=None,
                     num_steps=16, dc_lam=0.05, dc_iter=10, seed=0,
                     omega=1.0, t_start=1.0, warm_start=False, device="cuda"):
    """Physical observations and reconstructions use x_model = scale * x_phys."""
    mask = np.zeros(IMAGE_SIZE, dtype=bool)
    mask[PAD_LEFT:PAD_LEFT + MEASURED_DETECTORS] = True
    return dict(
        modality="ct", algorithm="pmf_dds", backend="nasn", device=str(device),
        inputs=dict(observation=observation),
        model=dict(geometry=dict(type="parallel_2d", domain_shape=[1, IMAGE_SIZE, IMAGE_SIZE],
            range_shape=[1, 900, IMAGE_SIZE],
            angles_rad=np.linspace(0, np.pi, 900, endpoint=False, dtype=np.float32).tolist(),
            center=PADDED_CENTER, detector_mask=mask.tolist()),
            measurement=dict(kind="line_integral", observation_model="line_integral")),
        params=dict(weights=str(Path(weights).resolve()),
            model_config=str(Path(model_config).resolve()) if model_config else None,
            num_inference_steps=num_steps, dc_lam=dc_lam, dc_iter=dc_iter,
            seed=seed, omega=omega, eta=0.0, image_scale=1 / float(scale),
            image_offset=0.0, t_start=t_start, warm_start=warm_start))


def reconstruct_slices(weights, sinogram_dir, scale, output, slices,
                       model_config=None, num_steps=16, dc_lam=0.05,
                       dc_iter=10, seed=0, omega=1.0, t_start=1.0,
                       warm_start=False, device="cuda"):
    from inv_framework.runner import run_experiment

    measurement = TruncatedMeasurement(sinogram_dir, scale=1.0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(configuration=dict(weights=str(Path(weights).resolve()), scale=float(scale),
                  image_units="physical; TIFF additionally saved in training scale"), slices={})
    for label in map(int, slices):
        observation = measurement.load(label, device="cpu").numpy()
        config = slice_experiment(weights, observation, scale, model_config=model_config,
            num_steps=num_steps, dc_lam=dc_lam, dc_iter=dc_iter, seed=seed,
            omega=omega, t_start=t_start, warm_start=warm_start, device=device)
        result = run_experiment(config, output_dir=output / f"slice_{label:06d}")
        report["slices"][str(label)] = dict(status=result.status, metrics=result.metrics,
            runtime_sec=result.runtime_sec, error=result.error)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        if result.status != "success":
            raise RuntimeError(f"NaSn slice {label} failed: {result.error}")
        image = result.outputs["reconstruction"][0, 0]
        tifffile.imwrite(output / f"recon_slice_{label:06d}.tif", (image * scale).astype(np.float32))
    return report
