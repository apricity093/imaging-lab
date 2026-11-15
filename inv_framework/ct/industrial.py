"""Industrial CT geometry construction and measured detector selection."""
import torch
from inv_framework.core.operator import LinearOperator


class DetectorMaskedOperator(LinearOperator):
    """Apply an explicitly declared detector mask to both sides of A."""

    nonnegative_matrix = True

    def __init__(self, operator, mask):
        self.operator = operator
        self.mask = mask
        self.domain_shape = operator.domain_shape
        self.range_shape = operator.range_shape
        self.num_angles = operator.num_angles

    def forward(self, x):
        return self.operator.forward(x) * self.mask.to(x)

    def adjoint(self, y):
        return self.operator.adjoint(y * self.mask.to(y))

    def fbp(self, y):
        from inv_framework.solvers.analytic import fbp
        return fbp(self.operator, y * self.mask.to(y))

    def pseudo_inverse(self, y):
        return self.operator.pseudo_inverse(y * self.mask.to(y))

    def fdk(self, y, **kwargs):
        return self.operator.fdk(y * self.mask.to(y), **kwargs)

    def close(self):
        close = getattr(self.operator, 'close', None)
        if close:
            close()


def build_industrial_operator(geometry, device):
    """Build ASTRA geometry in physical units, preserving non-cubic volumes."""
    from inv_framework.ct.backends.astra_cuda import (
        fan_beam_geometry, circular_cone_geometry, parallel_beam_3d_geometry,
        FanBeamASTRA2D, ConeBeamASTRA, ParallelBeamASTRA3D,
    )
    kind = geometry['type']
    domain, observed = geometry['domain_shape'], geometry['range_shape']
    if geometry.get('voxel_size', 1.0) <= 0:
        raise ValueError('voxel_size must be positive')
    pixel = geometry.get('det_pixel')
    pitch = torch.as_tensor(1.0 if pixel is None else pixel)
    if not bool(torch.isfinite(pitch).all() and (pitch > 0).all()):
        raise ValueError('det_pixel must be positive and finite')
    common = dict(angles=geometry['angles_rad'],
                  voxel_size=geometry.get('voxel_size', 1.0),
                  det_pixel=geometry.get('det_pixel'),
                  det_offset_u=geometry.get('det_offset_u', 0.0))
    if kind == 'fan_2d':
        if len(domain) != 3 or domain[0] != 1 or domain[1] != domain[2]:
            raise ValueError('fan_2d requires a single-channel square image')
        native = fan_beam_geometry(domain[-1], observed[-1],
            geometry['source_origin_distance'],
            geometry['source_origin_distance'] + geometry['origin_detector_distance'], **common)
        constructor = FanBeamASTRA2D
    else:
        builder = circular_cone_geometry if kind == 'cone_3d' else parallel_beam_3d_geometry
        distances = dict(sod=geometry['source_origin_distance'],
                         sdd=geometry['source_origin_distance'] + geometry['origin_detector_distance']) if kind == 'cone_3d' else {}
        native = builder(domain, (observed[0], observed[-1]), **distances, **common)
        constructor = ConeBeamASTRA if kind == 'cone_3d' else ParallelBeamASTRA3D
    if 'vectors' in geometry:
        import numpy as np
        native['vectors'] = np.asarray(geometry['vectors'], dtype=float)
    if 'window' in geometry:
        native['window'] = tuple(geometry['window'])
    return constructor(native, device=device)
