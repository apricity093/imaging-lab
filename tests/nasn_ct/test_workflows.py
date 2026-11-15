"""Small application checks, separate from released-model inference."""
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")
tifffile = pytest.importorskip("tifffile")

from inv_framework.applications.nasn_ct.data.preprocess import build_corpus, prepare_sinograms
from inv_framework.applications.nasn_ct.reconstruct import slice_experiment


def test_raw_preprocessing_resume_and_corpus(tmp_path):
    raw = tmp_path / 'raw' / 'Projection'
    raw.mkdir(parents=True)
    image = np.arange(48, dtype=np.float32).reshape(6, 8) + 40
    tifffile.imwrite(raw / 'flat_001.tif', image * 2)
    for index in range(3):
        tifffile.imwrite(raw / f'tomo_{index:03d}.tif', image + index)
    config = dict(prefixes=dict(tomo='tomo_', flat='flat_'),
                  tomo_range_1based=[1, 3], slice_range_1based=[2, 4],
                  detector_columns_1based=[1, 8],
                  phase_retrieval=dict(pixel_size_m=1.1e-6, distance_m=.147,
                                       energy_kev=27.4, epsilon=.01))
    destination = tmp_path / 'prepared'
    sino = prepare_sinograms(raw.parent, destination, config, device='cpu')
    before = tifffile.imread(sino / 'sino_000002.tif')
    assert before.shape == (3, 8) and np.isfinite(before).all()
    prepare_sinograms(raw.parent, destination, config, device='cpu')
    np.testing.assert_array_equal(tifffile.imread(sino / 'sino_000002.tif'), before)
    with pytest.raises(RuntimeError, match='stale work'):
        prepare_sinograms(raw.parent, destination, {**config, 'rivers_width': 3}, device='cpu')
    scans = []
    for index in range(4):
        scan = tmp_path / f'scan{index}' / 'recon_gridrec'
        scan.mkdir(parents=True)
        tifffile.imwrite(scan / 'slice_000001.tif', image + index * 20)
        scans.append(scan)
    stats = build_corpus(scans[:3], scans[3:], tmp_path / 'corpus', size=16)
    assert stats['final_scale'] > 0
    assert len(list((tmp_path / 'corpus/train').glob('*.tif'))) == 3
    assert len(list((tmp_path / 'corpus/val').glob('*.tif'))) == 1


@pytest.mark.parametrize('noise_scale', [None, 4.0])
def test_short_training_resume_export(tmp_path, monkeypatch, noise_scale):
    pytest.importorskip('safetensors')
    from inv_framework.solvers.deep.models.pmf import pmfDiT, pixelMeanFlow
    from inv_framework.applications.nasn_ct.training.train import entrypoint
    from inv_framework.applications.nasn_ct.weights import export_ema, load_pmf

    # Exercise the actual pMF loss and optimizer with a reduced DiT fixture.
    monkeypatch.setattr(pmfDiT, 'pmfDiT_B_128', partial(
        pmfDiT.pmfDiT, input_size=16, depth=2, hidden_size=32,
        patch_size=2, num_heads=4, aux_head_depth=1))
    torch.set_num_threads(2)
    corpus = tmp_path / 'corpus/train'
    corpus.mkdir(parents=True)
    for index in range(2):
        tifffile.imwrite(corpus / f'{index:06d}.tif', np.random.default_rng(index).random((16, 16)).astype(np.float32))
    config = dict(model_str='pmfDiT_B_128', img_size=16, lpips_lambda=0.0)
    if noise_scale is not None:
        config['noise_scale'] = noise_scale
    expected_noise_scale = 8.0 if noise_scale is None else noise_scale
    model_config = tmp_path / 'model.json'
    model_config.write_text(json.dumps(config))
    torch.manual_seed(42)
    initial = pixelMeanFlow(**dict(config, img_channels=1, num_classes=1,
                                  cfg_max=0, class_dropout_prob=0)).state_dict()
    common = ['--data-root', str(corpus.parent), '--workdir', str(tmp_path / 'training'),
              '--model-config', str(model_config), '--device', 'cpu', '--num-workers', '0',
              '--batch-size', '2', '--checkpoint-every', '1']
    entrypoint(common + ['--epochs', '1', '--from-scratch'])
    first = torch.load(tmp_path / 'training/ckpt_epoch_0.pt', weights_only=False)
    assert first['step'] == 1
    assert first['model_config']['noise_scale'] == expected_noise_scale
    assert any(not torch.equal(value, first['model'][key]) for key, value in initial.items())
    entrypoint(common + ['--epochs', '2', '--resume', str(tmp_path / 'training/ckpt_epoch_0.pt')])
    second = torch.load(tmp_path / 'training/ckpt_epoch_1.pt', weights_only=False)
    assert second['step'] == 2 and second['optimizer']['state']
    assert any(not torch.equal(value, second['ema'][key]) for key, value in first['ema'].items())
    weights, metadata = export_ema(tmp_path / 'training/ckpt_epoch_1.pt', tmp_path / 'ema.safetensors')
    loaded = load_pmf(weights, device='cpu')
    assert json.loads(metadata.read_text())['model']['img_size'] == 16
    assert json.loads(metadata.read_text())['model']['noise_scale'] == expected_noise_scale
    assert loaded.noise_scale == expected_noise_scale
    for key, value in loaded.state_dict().items():
        if key != 'net.rope_freqs':
            torch.testing.assert_close(value, second['ema'][key], rtol=0, atol=0)
    if noise_scale is None:
        legacy_metadata = json.loads(metadata.read_text())
        del legacy_metadata['model']['noise_scale']
        legacy_config = tmp_path / 'legacy.json'
        legacy_config.write_text(json.dumps(legacy_metadata))
        assert load_pmf(weights, config_path=legacy_config, device='cpu').noise_scale == 8.0


def test_public_request_scaling_and_warm_start(monkeypatch):
    from inv_framework.solvers.deep import registry
    from inv_framework.solvers.deep.registry import _build_pmf_dds
    from inv_framework.core.operator import LinearOperator

    class Identity(LinearOperator):
        domain_shape = range_shape = (1, 4, 4)
        def forward(self, x):
            return x
        def adjoint(self, y):
            return y
        def pseudo_inverse(self, y):
            return y

    class Prior:
        img_channels, img_size, noise_scale = 1, 4, 1
        def __call__(self, x, t):
            return torch.zeros_like(x)

    monkeypatch.setattr(registry, '_load_prior', lambda *args: Prior())
    solve = _build_pmf_dds(Identity(), dict(omega=1, num_inference_steps=1,
        eta=0, dc_lam=.05, dc_iter=2, warm_start=True, t_start=.5),
        weights='local', device='cpu', seed=0)
    assert torch.isfinite(solve(torch.ones(1, 1, 4, 4))).all()
    request = slice_experiment('weights.safetensors', np.zeros((1, 1, 900, 2048)), 3)
    assert request['params']['image_scale'] == 1 / 3
    assert sum(request['model']['geometry']['detector_mask']) == 410
    assert 'evaluation' not in request


def test_base_import_and_cli_are_lazy():
    code = """
import sys
import inv_framework.ct.api
from inv_framework.applications.nasn_ct.cli import build_parser
build_parser()
assert not any(name.startswith(('cv2', 'lpips', 'inv_framework.solvers.deep.models.pmf',
 'inv_framework.ct.backends.radon_cuda')) for name in sys.modules)
"""
    subprocess.run([sys.executable, '-c', code], check=True)


def test_slice_outputs_and_failure_summary(tmp_path, monkeypatch):
    from inv_framework.applications.nasn_ct import reconstruct
    from inv_framework.experiments import ExperimentResult
    from inv_framework import runner

    monkeypatch.setattr(reconstruct.TruncatedMeasurement, 'load',
                        lambda *args, **kwargs: torch.ones(1, 1, 900, 2048))
    requests = []
    def run(config, output_dir):
        requests.append(config)
        return ExperimentResult('success', 'ct', 'pmf_dds',
                                outputs={'reconstruction': np.ones((1, 1, 4, 4), dtype=np.float32)})
    monkeypatch.setattr(runner, 'run_experiment', run)
    reconstruct.reconstruct_slices('weights', tmp_path, 3, tmp_path / 'success', [1150], device='cpu')
    assert requests[0]['params']['image_scale'] == 1 / 3
    np.testing.assert_array_equal(tifffile.imread(tmp_path / 'success/recon_slice_001150.tif'), 3)
    monkeypatch.setattr(runner, 'run_experiment', lambda *args, **kwargs:
                        ExperimentResult('failed', 'ct', 'pmf_dds', error={'message': 'missing weights'}))
    with pytest.raises(RuntimeError, match='missing weights'):
        reconstruct.reconstruct_slices('weights', tmp_path, 3, tmp_path / 'failure', [1150])
    assert json.loads((tmp_path / 'failure/summary.json').read_text())['slices']['1150']['status'] == 'failed'
