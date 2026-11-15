"""Read the source industrial CT NPZ datasets as public experiment requests."""
from pathlib import Path
import numpy as np
import yaml


def toy_experiment(path, algorithm='sirt', *, device=None, params=None):
    """Keep line integrals in physical units and truth in evaluation only."""
    with np.load(Path(path), allow_pickle=False) as data:
        image, sino = data['image'], data['sinogram']
        if algorithm == 'cg_tikhonov':
            algorithm = 'cg'
            params = {'lam_rel': 1e-3, **(params or {})}
        if algorithm == 'fbp' and str(data['operator']) == 'astra_cone':
            algorithm = 'fdk'
        from inv_framework.experiments import normalize_ct_parameters
        presets = Path(__file__).resolve().parents[2] / 'configs/ct/industrial_baselines.yaml'
        defaults = yaml.safe_load(presets.read_text(encoding='utf-8')).get(algorithm, {})
        from inv_framework.solvers.deep.registry import ALGORITHMS
        scaling = {}
        if algorithm in ALGORITHMS:
            scaling = {key: float(data[key]) for key in ('image_scale', 'image_offset') if key in data}
        parameters = {**defaults, **scaling, **normalize_ct_parameters(params or {}, algorithm)}
        operator = str(data['operator'])
        kind = {'astra_cone': 'cone_3d', 'astra_parallel3d': 'parallel_3d',
                'astra_fan': 'fan_2d'}.get(operator, 'parallel_2d')
        dimension = 3 if kind in {'cone_3d', 'parallel_3d'} else 2
        domain = image.shape if dimension == 3 else (1, *image.shape)
        observed = sino[None] if dimension == 3 else sino[None, None]
        geometry = dict(type=kind, domain_shape=list(domain),
                        range_shape=list(observed.shape[1:]), angles_rad=data['theta'].tolist())
        if operator.startswith('astra_'):
            for key in ('voxel_size', 'det_pixel', 'det_offset_u'):
                geometry[key] = data[key].tolist()
            if kind != 'parallel_3d':
                geometry['source_origin_distance'] = float(data['sod'])
                geometry['origin_detector_distance'] = float(data['sdd'] - data['sod'])
        if 'center' in data:
            geometry['center'] = float(data['center'])
        if 'det_mask' in data:
            geometry['detector_mask'] = data['det_mask'].tolist()
        backend = 'astra' if operator.startswith('astra_') else 'nasn' if operator == 'tomopy_cuda' else 'torch'
        reference = image[None] if dimension == 3 else image[None, None]
        return dict(modality='ct', algorithm=algorithm, backend=backend,
                    device=device or ('cpu' if backend == 'torch' else 'cuda'),
                    params=parameters, inputs=dict(observation=observed),
                    model=dict(geometry=geometry, measurement=dict(kind='line_integral', observation_model='line_integral')),
                    evaluation=dict(reference=reference, data_range=float(image.max() - image.min())))
