"""CT execution of source deep priors in explicitly declared model units."""
from pathlib import Path
import importlib.util
import torch

from inv_framework.core.operator import LinearOperator
from inv_framework.solvers.base import InverseProblemSolver, SolveResult
from inv_framework.solvers.deep.registry import ALGORITHMS, build


def outer_parameter(name):
    return 'epochs' if name == 'dmplug' else 'num_iterations' if name in {'dip', 'inr'} else 'num_inference_steps'


def resource_paths(name, weights, model_config=None):
    path = Path(weights)
    if name.startswith('pmf_'):
        return [path, Path(model_config) if model_config else path.with_suffix('.json')]
    paths = [path / 'unet' / 'config.json', path / 'scheduler' / 'scheduler_config.json']
    if name in {'psld', 'resample', 'diffstategrad'}:
        paths.append(path / 'vqvae' / 'config.json')
    return paths


class _CountedModel:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self


class DeepCTSolver(InverseProblemSolver):
    """Keep physical/model scaling at the CT boundary and native loops intact."""

    def __init__(self, name, **parameters):
        self.name = name
        self.parameters = parameters

    def solve(self, measurement, operator, **kwargs):
        return self.solve_detailed(measurement, operator, **kwargs).reconstruction

    def solve_detailed(self, measurement, operator, *, control=None, **kwargs):
        from inv_framework.api.execution import BackendUnavailable
        options = dict(self.parameters)
        weights = options.pop('weights')
        seed = options.pop('seed')
        scale, offset = options.pop('image_scale'), options.pop('image_offset')
        budget = outer_parameter(self.name)
        if self.name == 'pmf_dds':
            warm, start = options['warm_start'], options['t_start']
            if (warm and not 0 < start < 1) or (not warm and start != 1):
                raise ValueError('warm_start requires 0 < t_start < 1; cold start requires t_start=1')
        if control is not None and control.max_iterations is not None:
            options[budget] = min(options[budget], control.max_iterations)
        if ALGORITHMS[self.name].learned:
            if not weights or not Path(weights).exists():
                raise BackendUnavailable(f'{self.name} requires local {ALGORITHMS[self.name].weights} weights')
            missing = [str(path) for path in resource_paths(self.name, weights, options.get('model_config')) if not path.is_file()]
            if missing:
                raise BackendUnavailable(f'Missing model resources: {missing}')
            dependency = 'safetensors' if self.name.startswith('pmf_') else 'diffusers'
            if importlib.util.find_spec(dependency) is None:
                raise BackendUnavailable(f'{self.name} requires {dependency}')
        if measurement.dtype != torch.float32:
            raise ValueError('CT deep model execution requires float32 observations')
        if len(operator.domain_shape) != 3:
            raise ValueError('CT deep priors require a 2D (C,H,W) domain')
        if self.name == 'budgeted_dps':
            from inv_framework.core.budget import CallableSubsetForwardOperator
            from inv_framework.solvers.subset import make_subset_operator
            from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
            if not isinstance(operator, ParallelBeamRadon2D):
                raise ValueError('budgeted_dps CT mapping requires unmasked torch parallel_2d source subsets')
            base = operator
            operator = CallableSubsetForwardOperator(
                lambda x, indices: make_subset_operator(base, torch.as_tensor(indices, device=base.angles.device)).forward(x),
                base.domain_shape, base.range_shape, base.num_angles, source_axis=2)
        shift = operator.forward(torch.full((measurement.shape[0], *operator.domain_shape), offset,
                                            device=measurement.device, dtype=measurement.dtype))
        model_measurement = (measurement - shift) / scale
        devices = [measurement.device.index or 0] if measurement.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.manual_seed(seed)
            solve = build(self.name, operator, options, weights=weights,
                          device=measurement.device, seed=seed)
            native = solve.solver
            counted = None
            if hasattr(native, 'model') and self.name not in {'dip', 'inr'}:
                counted = _CountedModel(native.model)
                native.model = counted
            reconstruction = (solve(model_measurement).detach() * scale + offset)
        predicted = operator.forward(reconstruction).detach()
        residual = float(torch.linalg.vector_norm(predicted - measurement) /
                         torch.linalg.vector_norm(measurement).clamp_min(1e-12))
        iterations = options[budget]
        metadata = dict(parameters={**self.parameters, **options}, native_termination_managed=True,
                        trajectory_available=False, image_scale=scale, image_offset=offset,
                        network_evaluations=counted.calls if counted else native.network_evaluations,
                        per_sample_fits=measurement.shape[0] if self.name == 'inr' else 1,
                        sampling_steps=(options['num_inference_steps'] *
                                        (iterations if self.name == 'dmplug' else 1)) if counted else None,
                        adaptation_updates=iterations * options['adapt_K'] if self.name == 'pmf_lora_dds' else 0)
        return SolveResult(reconstruction=reconstruction, predicted_measurement=predicted,
                           actual_iterations=iterations, status='max_iterations',
                           stopping_reason='fixed_budget_completed', final_residual=residual,
                           metadata=metadata)


_POSITIVE_INTEGER = dict(type='int', minimum=1)
_POSITIVE_REAL = dict(type='float', minimum=0, inclusive_minimum=False)
_NONNEGATIVE_REAL = dict(type='float', minimum=0, inclusive_minimum=True)
_PARAMETER_RULES = {
    'num_iterations': _POSITIVE_INTEGER,
    'num_inference_steps': _POSITIVE_INTEGER,
    'epochs': _POSITIVE_INTEGER,
    'input_channels': _POSITIVE_INTEGER,
    'hidden_features': _POSITIVE_INTEGER,
    'hidden_layers': _POSITIVE_INTEGER,
    'sirt_iterations': _POSITIVE_INTEGER,
    'cg_inner': _POSITIVE_INTEGER,
    'dc_iter': _POSITIVE_INTEGER,
    'lora_rank': _POSITIVE_INTEGER,
    'adapt_K': _POSITIVE_INTEGER,
    'inter_timesteps': _POSITIVE_INTEGER,
    'period': _POSITIVE_INTEGER,
    'pixel_maxiters': _POSITIVE_INTEGER,
    'latent_maxiters': _POSITIVE_INTEGER,
    'lr': _POSITIVE_REAL,
    'adapt_lr': _POSITIVE_REAL,
    'pixel_lr': _POSITIVE_REAL,
    'latent_lr': _POSITIVE_REAL,
    'scale': _POSITIVE_REAL,
    'dc_lam': _POSITIVE_REAL,
    'omega': _POSITIVE_REAL,
    'gamma': _NONNEGATIVE_REAL,
    'sigma': _NONNEGATIVE_REAL,
    'loss_measurement_weight': _NONNEGATIVE_REAL,
    'loss_noise_weight': _NONNEGATIVE_REAL,
    'eta': dict(type='float', minimum=0, maximum=1),
    'ema_decay': dict(type='float', nullable=True, minimum=0, maximum=1, inclusive_maximum=False),
    'beta': dict(type='float', minimum=0, maximum=1, inclusive_maximum=False),
    'var_cutoff': dict(type='float', minimum=0, inclusive_minimum=False, maximum=1),
    'schedule_actions': dict(type='subset_schedule', nullable=True),
    'key_indices': dict(type='int_array', nullable=True),
}
_ALGORITHM_PARAMETER_RULES = {
    'dps': {'num_inference_steps': dict(maximum=1000)},
    'mcg': {'num_inference_steps': dict(maximum=1000)},
    'psld': {'num_inference_steps': dict(maximum=1000), 'omega': _NONNEGATIVE_REAL},
    'pmf_dds': {'num_inference_steps': dict(maximum=20)},
    'pmf_lora_dds': {'num_inference_steps': dict(maximum=20)},
    'budgeted_dps': {'scale': _NONNEGATIVE_REAL},
}


def register_specs(algorithm, parameter):
    """Map source card defaults into the existing CT parameter contract."""
    specs = {}
    for name, source in ALGORITHMS.items():
        parameters = []
        for p in source.params:
            rules = {**_PARAMETER_RULES[p.name], **_ALGORITHM_PARAMETER_RULES.get(name, {}).get(p.name, {})}
            parameters.append(parameter(p.name, default=p.default, **rules, description=p.help,
                theoretical_constraint=p.constraint, theoretical_derivation=p.source))
        parameters += [parameter('weights', 'str', default=None, nullable=True, description='Local pretrained model directory or pMF safetensors file with adjacent JSON'),
                       parameter('seed', 'int', default=0, minimum=0, description='Seed for network initialization and sampling random streams'),
                       parameter('image_scale', 'float', default=1.0, minimum=0, inclusive_minimum=False, description='Positive physical image scale in x_phys = image_scale*x_model + image_offset'),
                       parameter('image_offset', 'float', default=0.0, description='Physical image offset, included in observation conversion through A(offset*1)')]
        if name.startswith('pmf_'):
            parameters.append(parameter('model_config', 'str', default=None, nullable=True, description='Explicit local pMF JSON configuration'))
        if name == 'pmf_dds':
            parameters += [parameter('t_start', 'float', default=1.0, minimum=0, inclusive_minimum=False, maximum=1),
                           parameter('warm_start', 'bool', default=False)]
        specs[name] = algorithm(name, source.title, (2,), ('parallel_2d', 'fan_2d'),
            source.summary, tuple(parameters), family='deep', convex=False,
            objective=source.summary, convergence_criteria=('fixed_native_budget',),
            initialization='random_network' if name in {'dip', 'inr'} else 'random_noise',
            failure_modes=('model_resource_mismatch', 'prior_distribution_mismatch', 'non_finite_optimization'),
            requirements=('2D differentiable operator', source.weights or 'no pretrained weights'),
            metadata=dict(source_commit='dc181fb65afbf300a9935c3c770231c44ffc0c78',
                          scope=source.scope, paper=source.paper, code=source.code,
                          outer_parameter=outer_parameter(name)))
    return specs


def describe_execution(record, context=None):
    """Describe local model prerequisites without constructing a model."""
    context = context or {}
    name = record['name']
    source = ALGORITHMS[name]
    params = {**record['default_parameters'], **context.get('params', {})}
    native = record['controls']['native']
    native.update(iteration_parameter=outer_parameter(name),
                  iteration_scope='per sample Adam iterations' if name == 'inr' else 'outer native loop',
                  stopping_criterion='Fixed native budget; no threshold convergence claim',
                  inner_parameters=[p for p in params if p in {'cg_inner','dc_iter','adapt_K','sirt_iterations','pixel_maxiters','latent_maxiters'}])
    record['controls']['unified_entry'].update(stopping_policy=False,
        note='Use the declared native outer-budget parameter; source loops use fixed budgets')
    record['weights'] = dict(required=source.learned, parameter='weights', resource=source.weights,
                             condition='Local resources, matching grid/channels and prediction type',
                             layout=('safetensors_with_adjacent_json' if name.startswith('pmf_') else
                                     'unet_scheduler_vqvae_directory' if name in {'psld','resample','diffstategrad'} else
                                     'unet_scheduler_directory'),
                             configuration_parameter='model_config' if name.startswith('pmf_') else None,
                             path_base='request configuration directory')
    status = record['runtime_status']
    dependency = 'safetensors' if name.startswith('pmf_') else 'diffusers'
    if source.learned:
        installed = importlib.util.find_spec(dependency) is not None
        record['dependencies']['required'][dependency] = installed
        if not installed:
            status['reasons'].append(f'Missing dependency: {dependency}')
        weights = params.get('weights')
        if not weights or not Path(weights).exists():
            status['reasons'].append(f'params.weights: local {source.weights} resources required')
        else:
            status['reasons'].extend(f'Missing model resource: {path}' for path in resource_paths(name, weights, params.get('model_config')) if not path.is_file())
            status['unresolved_conditions'].append('Model grid, channels, scheduler prediction and codec compatibility')
    geometry = context.get('model', {}).get('geometry', {})
    if source.learned and params.get('weights') and Path(params['weights']).exists():
        status['reasons'].extend(model_conditions(name, params, geometry))
    if geometry.get('type') and geometry['type'] not in record['geometry_types']:
        status['reasons'].append('Deep CT priors support parallel_2d and fan_2d')
    if name == 'budgeted_dps' and not params.get('schedule_actions'):
        status['reasons'].append('params.schedule_actions: explicit source-subset schedule required')
    if name == 'budgeted_dps' and (context.get('backend', 'torch') != 'torch' or geometry.get('detector_mask') is not None or geometry.get('type', 'parallel_2d') != 'parallel_2d'):
        status['reasons'].append('CT source subsets require unmasked torch parallel_2d')
    if status['reasons']:
        status['state'] = 'unavailable'
        record['runnable'] = False
    record['dependency_installed'] = all(record['dependencies']['required'].values())


def model_conditions(name, params, geometry):
    """Screen local JSON declarations without loading tensors or model classes."""
    import json
    weights = Path(params['weights'])
    pmf = name.startswith('pmf_')
    config = Path(params.get('model_config') or weights.with_suffix('.json')) if pmf else weights / 'unet/config.json'
    if not config.is_file():
        return []
    try:
        data = json.loads(config.read_text(encoding='utf-8'))
    except (ValueError, OSError) as error:
        return [f'Invalid model configuration {config}: {error}']
    if pmf:
        from inv_framework.solvers.deep.models.weights import DEFAULT_MODEL_CONFIG
        data = {**DEFAULT_MODEL_CONFIG, **data.get('model', {})}
    size = data.get('img_size' if pmf else 'sample_size')
    channels = data.get('img_channels' if pmf else 'in_channels')
    latent = name in {'psld', 'resample', 'diffstategrad'}
    domain = geometry.get('domain_shape')
    reasons = []
    if domain and not latent:
        grid = [size, size] if isinstance(size, int) else size
        if grid and list(domain[-2:]) != list(grid):
            reasons.append(f'Model grid {grid} differs from domain {domain}')
        if channels and domain[0] != channels:
            reasons.append(f'Model channels {channels} differ from domain {domain}')
    if not pmf:
        folders = ['unet', 'vqvae'] if latent else ['unet']
        for folder in folders:
            if not any((weights / folder).glob('diffusion_pytorch_model.*')):
                reasons.append(f'Missing local model tensors: {weights / folder}')
        codec_config = weights / 'vqvae/config.json'
        if latent and domain and codec_config.is_file():
            try:
                codec = json.loads(codec_config.read_text(encoding='utf-8'))
                if codec.get('in_channels') != domain[0]:
                    reasons.append('VQ codec channels differ from physical domain')
                if (codec.get('vq_embed_dim') or codec.get('latent_channels')) != channels:
                    reasons.append('VQ latent channels differ from UNet channels')
                grid = [size, size] if isinstance(size, int) else size
                if grid:
                    blocks = codec.get('block_out_channels', [64])
                    factor = 2 ** (len(blocks) - 1)
                    decoded = [int(axis) * factor for axis in grid]
                    if list(domain[-2:]) != decoded:
                        reasons.append(f'VQ decoded grid {decoded} differs from domain {domain}')
            except (ValueError, OSError) as error:
                reasons.append(f'Invalid codec configuration: {error}')
        scheduler = weights / 'scheduler/scheduler_config.json'
        if scheduler.is_file():
            try:
                prediction = json.loads(scheduler.read_text(encoding='utf-8')).get('prediction_type', 'epsilon')
                if prediction != 'epsilon':
                    reasons.append(f'CT diffusion requires epsilon prediction, received {prediction}')
            except (ValueError, OSError) as error:
                reasons.append(f'Invalid scheduler configuration: {error}')
    return reasons
