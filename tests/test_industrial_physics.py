"""Physical CT references from analytic Gaussian line integrals in world coordinates."""
import math
import numpy as np
import pytest
import torch

from inv_framework.ct.industrial import build_industrial_operator
from inv_framework.runner import run_experiment


def gaussian_reference(kind):
    """Integrate two off-axis Gaussian objects without invoking a projector.

    Coordinates use increasing X/Z and descending image Y. Each line integral
    is the closed-form integral of a multivariate Gaussian over physical length.
    Sources and detectors lie far outside the Gaussian support.
    """
    is_2d = kind == 'fan_2d'
    regular_cone = kind == 'regular_cone_3d'
    voxel = 1. if regular_cone else 2.
    shape = (1, 24, 24) if is_2d else (12, 24, 24)
    angles = np.arange(60) * (math.pi if kind == 'parallel_3d' else 2*math.pi) / 60
    window = [-20., 28., -26., 22.] if is_2d else [-20., 28., -26., 22., -10., 14.]
    if regular_cone:
        window = [-12., 12., -12., 12., -6., 6.]
    pitch = 2. if kind == 'parallel_3d' else 4.
    rows, cols = (1 if is_2d else 16), 40
    offset = 1.25
    if regular_cone:
        pitch, offset = 2., 0.
    geometry = dict(type=kind, domain_shape=list(shape), range_shape=[rows, len(angles), cols],
                    angles_rad=angles.tolist(), voxel_size=2.,
                    det_pixel=pitch if is_2d else [pitch, pitch], det_offset_u=offset,
                    source_origin_distance=160., origin_detector_distance=160., window=window)
    if regular_cone:
        geometry = dict(type='cone_3d', domain_shape=list(shape), range_shape=[rows, len(angles), cols],
            angles_rad=angles.tolist(), source_origin_distance=160., origin_detector_distance=160.,
            detector_spacing_row=pitch, detector_spacing_col=pitch, detector_rows=rows, detector_cols=cols)
    x = window[0] + (np.arange(shape[-1])+.5)*voxel
    y = window[3] - (np.arange(shape[-2])+.5)*voxel
    z = np.array([0.]) if is_2d else window[4] + (np.arange(shape[0])+.5)*voxel
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    points = np.stack([xx, yy, zz], -1)
    components = [(1., [-7., 4., -2.], [4., 5., 3.]),
                  (.65, [10., -10., 5.], [3., 4., 2.5])]
    if regular_cone:
        components = [(a, np.asarray(c)/2, np.asarray(s)/2) for a, c, s in components]
    if is_2d:
        components = [(a, [c[0], c[1], 0.], s) for a, c, s in components]
    truth = np.zeros(shape)
    for amplitude, center, sigma in components:
        truth += amplitude*np.exp(-.5*np.sum(((points-center)/sigma)**2, axis=-1))
    # Independently specify the scanner orbit and detector coordinates.
    angle = angles[None, :, None]
    u = (np.arange(cols)-(cols-1)/2+offset)[None, None, :]*pitch
    v = (np.arange(rows)-(rows-1)/2)[:, None, None]*pitch
    px = np.broadcast_to(np.cos(angle)*u, (rows, len(angles), cols))
    py = np.broadcast_to(np.sin(angle)*u, px.shape)
    pz = np.broadcast_to(v, px.shape)
    if kind == 'parallel_3d':
        origin = np.stack([px, py, pz], -1)
        direction = np.stack([np.broadcast_to(np.sin(angle), px.shape),
                              np.broadcast_to(-np.cos(angle), px.shape), np.zeros_like(px)], -1)
    else:
        origin = np.stack([np.broadcast_to(160*np.sin(angle), px.shape),
                           np.broadcast_to(-160*np.cos(angle), px.shape), np.zeros_like(px)], -1)
        detector = np.stack([px-160*np.sin(angle), py+160*np.cos(angle), pz], -1)
        direction = detector-origin
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    sino = np.zeros(px.shape)
    for amplitude, center, sigma in components:
        delta = origin-center
        quadratic = np.sum((direction/sigma)**2, axis=-1)
        linear = np.sum(delta*direction/np.square(sigma), axis=-1)
        constant = np.sum((delta/sigma)**2, axis=-1)
        sino += amplitude*np.sqrt(2*math.pi/quadratic)*np.exp(-.5*(constant-linear**2/quadratic))
    return geometry, truth, sino, points, components


def test_regular_cone_public_fdk_physics():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('astra')
    from inv_framework.api.execution import build_operator
    from inv_framework.api.schema import CTTaskRequest, load_request_case
    from inv_framework.ct.backends.astra_fdk_adapter import ASTRAFDKOperator3D
    geometry, truth, sino, points, components = gaussian_reference('regular_cone_3d')
    observed = np.stack([sino, .6*sino]).astype(np.float32)
    measurement = dict(kind='line_integral', observation_model='line_integral')
    request = CTTaskRequest('external', 'fdk', backend='astra', device='cuda',
        inputs=dict(observation=observed, geometry=geometry, measurement=measurement))
    case = load_request_case(request, 'cuda')
    operator = build_operator(case, 'cuda', 'astra')
    try:
        assert isinstance(operator, ASTRAFDKOperator3D)
        prediction = operator(torch.tensor(truth[None], dtype=torch.float32, device='cuda'))
        projection_error = np.linalg.norm(prediction.cpu().numpy()[0]-sino)/np.linalg.norm(sino)
        assert projection_error < .08
    finally:
        operator.close()
    result = run_experiment(dict(modality='ct', algorithm='fdk', backend='astra', device='cuda',
        inputs=dict(observation=observed), model=dict(geometry=geometry, measurement=measurement)))
    assert result.status == 'success', result.error
    reconstruction = result.outputs['reconstruction']
    assert reconstruction.shape == (2, *truth.shape)
    np.testing.assert_allclose(reconstruction[1], .6*reconstruction[0], rtol=1e-5, atol=1e-6)
    relative_error = np.linalg.norm(reconstruction[0]-truth)/np.linalg.norm(truth)
    assert relative_error < .15
    for _, center, sigma in components:
        roi = np.sum(((points-center)/sigma)**2, axis=-1) < 2.25
        mass = reconstruction[0][roi].sum()
        assert .85 < mass/truth[roi].sum() < 1.15
        centroid = (reconstruction[0][roi, None]*points[roi]).sum(0)/mass
        expected = (truth[roi, None]*points[roi]).sum(0)/truth[roi].sum()
        assert np.linalg.norm(centroid-expected) < .5
    print('regular_cone_3d fdk', 'projection_relative_error', projection_error,
          'image_relative_error', relative_error)


@pytest.mark.parametrize('kind', ['fan_2d', 'parallel_3d', 'cone_3d'])
def test_independent_gaussian_physics(kind):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('astra')
    geometry, truth, sino, points, components = gaussian_reference(kind)
    op = build_industrial_operator(geometry, 'cuda')
    try:
        prediction = op(torch.tensor(truth[None], dtype=torch.float32, device='cuda')).cpu().numpy()[0]
    finally:
        op.close()
    projection_error = np.linalg.norm(prediction-sino)/np.linalg.norm(sino)
    assert projection_error < .08
    for algorithm in ('fdk' if kind == 'cone_3d' else 'fbp', 'sirt'):
        config = dict(modality='ct', algorithm=algorithm, backend='astra', device='cuda',
            inputs=dict(observation=sino[None].astype(np.float32)),
            model=dict(geometry=geometry, measurement=dict(kind='line_integral', observation_model='line_integral')),
            params=dict(num_iterations=40, min_value=0.) if algorithm == 'sirt' else {})
        result = run_experiment(config)
        assert result.status == 'success', result.error
        reconstruction = result.outputs['reconstruction'][0]
        relative_error = np.linalg.norm(reconstruction-truth)/np.linalg.norm(truth)
        assert relative_error < .25, (kind, algorithm, relative_error)
        for amplitude, center, sigma in components:
            # Local mass and centroid retain absolute amplitude and signed position.
            roi = np.sum(((points-center)/sigma)**2, axis=-1) < 2.25
            mass = reconstruction[roi].sum()
            assert .75 < mass/truth[roi].sum() < 1.25
            centroid = (reconstruction[roi, None]*points[roi]).sum(0)/mass
            expected = (truth[roi, None]*points[roi]).sum(0)/truth[roi].sum()
            assert np.linalg.norm(centroid-expected) < 1.0
        print(kind, algorithm, 'projection_relative_error', projection_error,
              'image_relative_error', relative_error)


def test_parallel_fbp_uses_effective_window():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('astra')
    geometry, _, sino, _, _ = gaussian_reference('parallel_3d')
    observation = torch.tensor(sino[None], dtype=torch.float32, device='cuda')
    reconstructions = []
    for nominal_voxel in (2., 1.):
        op = build_industrial_operator({**geometry, 'voxel_size': nominal_voxel}, 'cuda')
        try:
            reconstructions.append(op.fbp(observation))
        finally:
            op.close()
    torch.testing.assert_close(*reconstructions, rtol=1e-5, atol=1e-6)
    anisotropic = {**geometry, 'window': [-20., 28., -26., 22., -10., 2.]}
    op = build_industrial_operator(anisotropic, 'cuda')
    try:
        with pytest.raises(ValueError, match='isotropic'):
            op.fbp(observation)
    finally:
        op.close()
