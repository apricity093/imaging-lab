"""Industrial geometry contracts through the public experiment entry point."""
import math
import numpy as np
import pytest
import torch
from inv_framework.ct.toy import toy_experiment
from inv_framework.runner import run_experiment
from inv_framework.ct.backends.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.algebraic import sirt, landweber, LandweberSolver


def test_classical_precision_and_automatic_step():
    op = ParallelBeamRadon2D(8, num_angles=6)
    y = op(torch.ones(1, 1, 8, 8, dtype=torch.float64))
    assert sirt(op, y, num_iterations=2).dtype == y.dtype
    plain = landweber(op, y, num_iterations=2)
    detailed = LandweberSolver(num_iterations=2).solve_detailed(y, op)
    torch.testing.assert_close(plain, detailed.reconstruction)


def test_source_truncation_and_relative_cg():
    config = toy_experiment('examples/ct_data/toy_truncated_ct_128.npz', 'cg_tikhonov',
                            params={'max_iter': 2})
    result = run_experiment(config)
    assert result.status == 'success', result.error
    assert result.effective_params['strength'] == pytest.approx(
        result.effective_params['lam_rel'] * result.effective_params['operator_norm_squared'])
    assert result.outputs['reconstruction'].shape == (1, 1, 128, 128)


@pytest.mark.parametrize('kind', ['fan_2d', 'parallel_3d', 'cone_3d'])
def test_industrial_adjoint_and_public_execution(kind):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('astra')
    from inv_framework.ct.industrial import build_industrial_operator
    domain = [1, 8, 8] if kind == 'fan_2d' else [3, 8, 6]
    geometry = dict(type=kind, domain_shape=domain,
        range_shape=[1 if kind == 'fan_2d' else 4, 8, 12],
        angles_rad=(np.arange(8)*2*math.pi/8).tolist(),
        source_origin_distance=30., origin_detector_distance=30.,
        det_pixel=2. if kind == 'fan_2d' else [2., 2.])
    op = build_industrial_operator(geometry, 'cuda')
    try:
        torch.manual_seed(7)
        x = torch.rand(1, *domain, device='cuda', requires_grad=True)
        y = torch.randn(1, *op.range_shape, device='cuda')
        projected = op(x)
        torch.testing.assert_close((projected*y).sum(), (x*op.adjoint(y)).sum(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(torch.autograd.grad((projected*y).sum(), x)[0], op.adjoint(y))
    finally:
        op.close()
    for algorithm in ('fdk' if kind == 'cone_3d' else 'fbp', 'sirt', 'landweber', 'cg'):
        config = dict(modality='ct', algorithm=algorithm, backend='astra', device='cuda',
            params={} if algorithm in {'fbp', 'fdk'} else dict(num_iterations=2),
            inputs=dict(observation=projected.detach().cpu().numpy()),
            model=dict(geometry=geometry, measurement=dict(kind='line_integral', observation_model='line_integral')))
        result = run_experiment(config)
        assert result.status == 'success', result.error
        assert result.outputs['reconstruction'].shape == (1, *domain)
