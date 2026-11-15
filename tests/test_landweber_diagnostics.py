"""Landweber step evidence and the NaSn nonnegative Schur-bound contract."""
import numpy as np
import pytest
import torch

from inv_framework.core.operator import LinearOperator
from inv_framework.core.fidelity import QuadraticFidelity
from inv_framework.solvers.quadratic import QuadraticSolver
from inv_framework.ct.backends.nasn_cuda import TomoPyCUDARadon2D
from inv_framework.runner import run_experiment


class PositiveDiagonal(LinearOperator):
    domain_shape = range_shape = (1, 2)
    nonnegative_matrix = True

    def forward(self, x):
        return x*x.new_tensor([2., 3.])

    def adjoint(self, y):
        return self.forward(y)


@pytest.mark.parametrize('mode', ['power', 'weighted', 'regularized', 'explicit'])
def test_landweber_step_evidence(mode):
    operator = PositiveDiagonal()
    observation = torch.ones(1, 1, 2, dtype=torch.float64)
    params = dict(num_iterations=1, tolerance=0., regularizer='none')
    if mode == 'regularized':
        params.update(regularizer='tikhonov', strength=.5)
    if mode == 'explicit':
        params['step_size'] = .05
    solver = QuadraticSolver('landweber', **params)
    if mode == 'weighted':
        solver.fidelity = QuadraticFidelity(observation, observation.new_tensor([[[1., 2.]]]))
    result = solver.solve_detailed(observation, operator)
    evidence = result.metadata
    basis = 'power_iteration_estimate' if mode == 'power' else 'explicit' if mode == 'explicit' else 'upper_bound'
    assert evidence['step_basis'] == basis
    if mode == 'power':
        assert evidence['step_norm_squared'] == pytest.approx(9., rel=1e-6)
        assert evidence['step_size'] == .9/evidence['step_norm_squared']
        assert evidence['step_estimator_iterations'] == 12
        assert evidence['step_safety_factor'] == .9
    elif mode != 'explicit':
        assert evidence['step_norm_squared'] == evidence['augmented_squared_norm_bound']
        assert evidence['step_size'] == 1/evidence['step_norm_squared']
    else:
        assert evidence['step_norm_squared'] is None
        assert evidence['step_size'] == .05
    weights = solver.fidelity.weights if solver.fidelity else torch.ones_like(observation)
    expected = evidence['step_size']*operator.adjoint(weights*observation)
    torch.testing.assert_close(result.reconstruction, expected)


def test_nasn_schur_bound_beyond_generic_size_limit(monkeypatch):
    """Substitute the compiled boundary, retaining the actual public NaSn path.

    A positive row-sum projector has a known squared norm of views*width.
    Native CUDA kernel execution remains a separate environment-dependent check.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from inv_framework.ct.backends import radon_cuda
    size, views = 65, 4

    def project(image, theta, center):
        return image.sum(-2)[:, None, :].expand(-1, len(theta), -1).contiguous()

    def backproject(sino, theta, center, height, width):
        return sino.sum(1)[:, None, :].expand(-1, height, -1).contiguous()

    monkeypatch.setattr(radon_cuda, 'project', project)
    monkeypatch.setattr(radon_cuda, 'bp', backproject)
    operator = TomoPyCUDARadon2D(size, num_angles=views, device='cuda')
    reference = torch.ones(1, 1, size, size, device='cuda')
    assert operator.nonnegative_matrix
    assert operator.squared_norm_upper_bound(reference) == views*size
    config = dict(modality='ct', algorithm='landweber', backend='nasn', device='cuda',
        params=dict(num_iterations=2), inputs=dict(observation=np.full((1, 1, views, size), size, dtype=np.float32)),
        model=dict(geometry=dict(type='parallel_2d', domain_shape=[1, size, size],
            range_shape=[1, views, size], angles_rad=operator.angles.cpu().tolist()),
            measurement=dict(kind='line_integral', observation_model='line_integral')))
    result = run_experiment(config)
    assert result.status == 'success', result.error
    evidence = result.diagnostics['convergence']['metadata']
    assert evidence['step_basis'] == 'power_iteration_estimate'
    assert evidence['step_norm_squared'] == pytest.approx(views*size)
    assert result.effective_params['step_size'] == pytest.approx(.9/(views*size))
    np.testing.assert_allclose(result.outputs['reconstruction'], .99, atol=1e-5)
