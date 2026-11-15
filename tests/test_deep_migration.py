"""Source deep-prior behavior and public CT execution."""
import torch
import pytest
from inv_framework.core.operator import LinearOperator
from inv_framework.solvers.deep.dip import DIPSolver
from inv_framework.solvers.deep.inr import INRSolver
from inv_framework.solvers.deep.cg import cg_data_consistency_grad
from inv_framework.solvers.deep.diffusion import DPSSolver, VPSchedule
from inv_framework.solvers.deep.registry import ALGORITHMS

class Diagonal(LinearOperator):
    domain_shape = range_shape = (1, 8, 8)
    def forward(self, x):
        return x * torch.linspace(.7, 1.3, 8, device=x.device)[None,None,None,:]
    adjoint = forward

def test_dip_ema_and_callback():
    torch.set_num_threads(1)
    values = []
    solver = DIPSolver(model_factory=lambda: torch.nn.Conv2d(2,1,1),
        input_channels=2, num_iterations=3, ema_decay=.8,
        callback=lambda i,x,loss: values.append(x.clone()))
    x = solver.solve(torch.ones(1,1,8,8),Diagonal())
    expected = values[0]
    for value in values[1:]:
        expected = .8*expected + .2*value
    torch.testing.assert_close(x,expected,rtol=0,atol=0)

def test_inr_fits_each_sample():
    factories = []
    def factory():
        model = torch.nn.Linear(2,1)
        factories.append(model)
        return model
    x = INRSolver(model_factory=factory,num_iterations=2).solve(torch.stack(
        [torch.ones(1,8,8),torch.zeros(1,8,8)]),Diagonal())
    assert len(factories)==2
    assert not torch.equal(x[0],x[1])

def test_dps_generator_and_step_context():
    from inv_framework.solvers.deep.diffusion import PosteriorSampling
    class Condition(PosteriorSampling):
        def apply(self,*args,**kw):
            steps.append((kw["step_index"],kw["num_steps"]))
            return super().apply(*args,**kw)
    steps=[]
    A=Diagonal()
    class Noise(torch.nn.Module):
        def forward(self,x,t):return .1*x
    solver=DPSSolver(Noise(),VPSchedule(num_train_timesteps=20),
        num_inference_steps=4,eta=1.,conditioning=Condition(A,scale=.1))
    first=solver.solve(torch.ones(1,1,8,8),A,generator=torch.Generator().manual_seed(7))
    torch.randn(50)
    second=solver.solve(torch.ones(1,1,8,8),A,generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert steps == [(i,4) for i in range(4)]*2

def test_cg_correction_matches_dense_and_differentiates():
    A=Diagonal()
    x=torch.randn(2,1,8,8,requires_grad=True)
    y=torch.ones_like(x)
    out=cg_data_consistency_grad(x,y,A,lam=.3,max_iter=20,tol=1e-9)
    weights=A.forward(torch.ones_like(x))
    expected=(weights*y+.3*x)/(weights.square()+.3)
    torch.testing.assert_close(out,expected,rtol=1e-5,atol=1e-6)
    gradient,=torch.autograd.grad(out.sum(),x)
    torch.testing.assert_close(gradient,.3/(weights.square()+.3),rtol=1e-4,atol=1e-5)

def test_physical_model_affine_conversion(monkeypatch):
    from types import SimpleNamespace
    from inv_framework.ct.deep import DeepCTSolver
    from inv_framework.solvers.deep.registry import defaults
    op = Diagonal()
    physical = torch.full((1,1,8,8), .7)
    captured = []
    def build(*args, **kwargs):
        def solve(y):
            captured.append(y)
            return (physical - .2) / .5
        solve.solver = SimpleNamespace(network_evaluations=1)
        return solve
    monkeypatch.setattr('inv_framework.ct.deep.build',build)
    solver = DeepCTSolver('inr', **defaults('inr'), weights=None, seed=0,
                          image_scale=.5, image_offset=.2)
    result = solver.solve_detailed(op(physical),op)
    torch.testing.assert_close(captured[0],op((physical-.2)/.5))
    torch.testing.assert_close(result.reconstruction,physical)

def test_public_inr_and_missing_weights(tmp_path):
    from inv_framework.runner import run_experiment
    config=dict(modality="ct",algorithm="inr",device="cpu",
        model=dict(geometry=dict(type="parallel_2d",domain_shape=[1,8,8],
            range_shape=[1,4,8],angles_rad=[0.,.5,1.,1.5]),
            measurement=dict(observation_domain="line_integral",observation_model="line_integral")),
        inputs=dict(observation=torch.ones(1,1,4,8).numpy()),
        params=dict(num_iterations=2,hidden_features=8,hidden_layers=1,image_scale=.5,image_offset=.2))
    result=run_experiment(config,output_dir=tmp_path/"inr")
    assert result.status=="success"
    assert result.iterations==2
    assert not result.converged
    assert (tmp_path/"inr"/"reconstruction.npy").exists()
    config.update(algorithm="dps",params=dict(num_inference_steps=2))
    result=run_experiment(config,output_dir=tmp_path/"dps")
    assert result.status!="success"
    assert "weights" in str(result.error)

def test_deep_catalog_and_outer_controls():
    from inv_framework.ct.api import list_capabilities
    records={r["name"]:r for r in list_capabilities()["algorithms"]}
    for name in set(ALGORITHMS)-{"budgeted_dps"}:
        assert records[name]["build_reachable"]
        assert records[name]["controls"]["native"]["iteration_parameter"] == (
            "epochs" if name=="dmplug" else "num_iterations" if name in {"dip","inr"} else "num_inference_steps")


@pytest.mark.parametrize('name, parameter, accepted, rejected', [
    ('dip', 'ema_decay', [None, 0., .99], [-.1, 1.]),
    ('reddiff', 'sigma', [0., .01], [-.01, None]),
    ('hybridreg', 'sigma', [0., .01], [-.01, None]),
    ('pmf_dds', 'num_inference_steps', [1, 20], [0, 21, None]),
    ('pmf_lora_dds', 'num_inference_steps', [1, 20], [0, 21, None]),
    ('dps', 'num_inference_steps', [1, 1000], [0, 1001]),
    ('mcg', 'num_inference_steps', [1, 1000], [0, 1001]),
    ('psld', 'num_inference_steps', [1, 1000], [0, 1001]),
    ('psld', 'omega', [0., 1.], [-.1]),
    ('hybridreg', 'beta', [0., .99], [-.1, 1.]),
    ('diffstategrad', 'var_cutoff', [.01, 1.], [0., 1.01]),
    ('pmf_dds', 'dc_lam', [.01, 5.], [0., -.1]),
])
def test_public_deep_parameter_boundaries(name, parameter, accepted, rejected):
    from inv_framework.api.schema import normalize_request
    from inv_framework.api.execution import ConfigError
    from inv_framework.ct.registry import SOLVER_SPECS
    spec = SOLVER_SPECS[name].parameter(parameter)
    for value in accepted:
        assert spec.normalize(value) == value
        _, _, params = normalize_request(dict(case_id='boundary', algorithm=name,
                                              params={parameter: value}))
        assert params[parameter] == value
    for value in rejected:
        with pytest.raises(ConfigError):
            normalize_request(dict(case_id='boundary', algorithm=name, params={parameter: value}))


def test_diffstategrad_public_inner_budgets(tmp_path, monkeypatch):
    pytest.importorskip('diffusers')
    import safetensors.torch
    from diffusers import UNet2DModel, VQModel, DDPMScheduler
    from inv_framework.runner import run_experiment
    from inv_framework.solvers.deep.diffusion import DiffStateGradSolver
    from inv_framework.experiments import execution_controls
    torch.set_num_threads(1)
    torch.manual_seed(7)
    weights = tmp_path / 'weights'
    net = UNet2DModel(sample_size=8, in_channels=1, out_channels=1, layers_per_block=1,
        block_out_channels=(8,), down_block_types=('DownBlock2D',),
        up_block_types=('UpBlock2D',), norm_num_groups=4)
    net.save_pretrained(weights / 'unet')
    codec = VQModel(in_channels=1, out_channels=1, latent_channels=1, vq_embed_dim=1,
        num_vq_embeddings=8, block_out_channels=(8,), norm_num_groups=4, sample_size=8,
        scaling_factor=1)
    codec.save_pretrained(weights / 'vqvae')
    DDPMScheduler(num_train_timesteps=80).save_pretrained(weights / 'scheduler')

    events = []
    active = []
    original_step = torch.optim.AdamW.step
    def step(optimizer, *args, **kwargs):
        active[-1]['updates'] += 1
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    def trace(original, kind):
        def optimization(self, *args, **kwargs):
            event = dict(kind=kind, limit=kwargs['max_iters'], updates=0)
            events.append(event)
            active.append(event)
            try:
                return original(self, *args, **kwargs)
            finally:
                active.pop()
        return optimization
    for kind in ('pixel', 'latent'):
        method = kind + '_optimization'
        monkeypatch.setattr(DiffStateGradSolver, method,
                            trace(getattr(DiffStateGradSolver, method), kind))

    params = dict(weights=str(weights), num_inference_steps=40, pixel_maxiters=3,
                  latent_maxiters=2, pixel_lr=1e-4, latent_lr=1e-4, seed=7)
    config = dict(modality='ct', algorithm='diffstategrad', device='cpu', params=params,
        model=dict(geometry=dict(type='parallel_2d', domain_shape=[1,8,8],
            range_shape=[1,4,8], angles_rad=[0.,.5,1.,1.5]),
            measurement=dict(kind='line_integral', observation_model='line_integral')),
        inputs=dict(observation=torch.ones(1,1,4,8).numpy()))
    result = run_experiment(config, output_dir=tmp_path / 'result')
    assert result.status == 'success', result.error
    assert result.iterations == 40
    assert result.effective_params['pixel_maxiters'] == 3
    assert result.effective_params['latent_maxiters'] == 2
    assert events == [dict(kind='pixel', limit=3, updates=3),
                      dict(kind='latent', limit=2, updates=2)]
    controls = execution_controls('ct', 'diffstategrad', result.effective_params)
    assert controls['stopping_policy']['max_iterations'] == 'num_inference_steps'

    from inv_framework.api.schema import normalize_request
    from inv_framework.api.execution import build_solver
    from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
    from inv_framework.solvers.base import SolveControl
    from types import SimpleNamespace
    _, name, normalized = normalize_request(dict(case_id='inner_budget',
        algorithm='diffstategrad', params={**params, 'num_inference_steps': 60}))
    operator = ParallelBeamRadon2D(8, angles=torch.tensor([0., .5, 1., 1.5]))
    measurement = torch.ones(1, 1, 4, 8)
    solver = build_solver(name, normalized, SimpleNamespace(measurement=measurement))
    events.clear()
    detailed = solver.solve_detailed(measurement, operator, control=SolveControl(max_iterations=40))
    assert detailed.actual_iterations == 40
    assert detailed.metadata['sampling_steps'] == 40
    assert detailed.metadata['network_evaluations'] > detailed.metadata['sampling_steps']
    assert detailed.metadata['parameters']['num_inference_steps'] == 40
    assert detailed.metadata['parameters']['pixel_maxiters'] == 3
    assert detailed.metadata['parameters']['latent_maxiters'] == 2
    assert events == [dict(kind='pixel', limit=3, updates=3),
                      dict(kind='latent', limit=2, updates=2)]
