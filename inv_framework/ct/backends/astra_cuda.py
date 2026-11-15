"""Industrial CT geometry and ASTRA analytic reconstruction with matched ray operators."""

import math

import numpy as np
import torch

from inv_framework.core.operator import LinearOperator


def circular_cone_geometry(volume_shape, det_shape, sod, sdd, voxel_size=1.0,
                           det_pixel=None, num_angles=360, angles=None,
                           det_offset_u=0.0):
    """Geometry dict for a circular cone-beam scan around the z axis.

    volume_shape : (nz, ny, nx) voxels, centred on the rotation axis.
    det_shape    : (det_rows, det_cols).
    sod, sdd     : source-object and source-detector distances.
    voxel_size   : isotropic voxel pitch.
    det_pixel    : (dv, du) detector pitch; default ``voxel_size * sdd / sod``
                   (one voxel at the axis maps to one detector pixel).
    angles       : source angles in radians; default ``num_angles`` uniform
                   angles over a full turn [0, 2 pi).
    det_offset_u : detector-centre shift along u, in detector pixels.
    """
    nz, ny, nx = (int(n) for n in volume_shape)
    rows, cols = (int(n) for n in det_shape)
    if not 0 < sod < sdd:
        raise ValueError("expected 0 < sod < sdd")
    magnification = sdd / sod
    if det_pixel is None:
        det_pixel = (voxel_size * magnification,) * 2
    dv, du = (float(p) for p in det_pixel)
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, int(num_angles), endpoint=False)
    angles = np.asarray(angles, dtype=np.float64)
    sin, cos, zero = np.sin(angles), np.cos(angles), np.zeros_like(angles)
    src = np.stack([sin * sod, -cos * sod, zero], 1)
    det = np.stack([-sin * (sdd - sod), cos * (sdd - sod), zero], 1)
    u = np.stack([cos * du, sin * du, zero], 1)
    v = np.stack([zero, zero, np.full_like(angles, dv)], 1)
    det = det + det_offset_u * u
    half = np.array([nx, ny, nz], dtype=np.float64) * voxel_size / 2
    return dict(kind="cone", volume_shape=(nz, ny, nx), det_shape=(rows, cols), sod=float(sod),
                sdd=float(sdd), voxel_size=float(voxel_size), det_pixel=(dv, du),
                angles=angles.astype(np.float32), det_offset_u=float(det_offset_u),
                vectors=np.concatenate([src, det, u, v], 1),
                window=(-half[0], half[0], -half[1], half[1], -half[2], half[2]))


def parallel_beam_3d_geometry(volume_shape, det_shape, voxel_size=1.0, det_pixel=None,
                              num_angles=180, angles=None, det_offset_u=0.0):
    """Geometry dict for a synchrotron-style parallel-beam scan around the z axis.

    volume_shape : (nz, ny, nx) voxels, centred on the rotation axis.
    det_shape    : (det_rows, det_cols).
    det_pixel    : (dv, du) detector pitch; default ``voxel_size``.
    angles       : projection angles in radians; default ``num_angles``
                   uniform angles over [0, pi).
    det_offset_u : rotation-axis offset on the detector along u, in pixels.
    """
    nz, ny, nx = (int(n) for n in volume_shape)
    rows, cols = (int(n) for n in det_shape)
    if det_pixel is None:
        det_pixel = (voxel_size, voxel_size)
    dv, du = (float(p) for p in det_pixel)
    if angles is None:
        angles = np.linspace(0, np.pi, int(num_angles), endpoint=False)
    angles = np.asarray(angles, dtype=np.float64)
    sin, cos, zero = np.sin(angles), np.cos(angles), np.zeros_like(angles)
    ray = np.stack([sin, -cos, zero], 1)          # astra.geom_2vec('parallel3d') convention
    u = np.stack([cos * du, sin * du, zero], 1)
    v = np.stack([zero, zero, np.full_like(angles, dv)], 1)
    det = det_offset_u * u
    half = np.array([nx, ny, nz], dtype=np.float64) * voxel_size / 2
    return dict(kind="parallel3d", volume_shape=(nz, ny, nx), det_shape=(rows, cols),
                voxel_size=float(voxel_size), det_pixel=(dv, du),
                angles=angles.astype(np.float32), det_offset_u=float(det_offset_u),
                vectors=np.concatenate([ray, det, u, v], 1),
                window=(-half[0], half[0], -half[1], half[1], -half[2], half[2]))


def fan_beam_geometry(image_size, det_cols, sod, sdd, voxel_size=1.0, det_pixel=None,
                      num_angles=360, angles=None, det_offset_u=0.0):
    """Geometry dict for the fan-beam mid-plane of ``circular_cone_geometry``.

    Same conventions (source at angle 0 on -y, detector u along +x); the
    detector pitch defaults to ``voxel_size * sdd / sod``.
    """
    if not 0 < sod < sdd:
        raise ValueError("expected 0 < sod < sdd")
    du = voxel_size * sdd / sod if det_pixel is None else float(det_pixel)
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, int(num_angles), endpoint=False)
    angles = np.asarray(angles, dtype=np.float64)
    sin, cos = np.sin(angles), np.cos(angles)
    src = np.stack([sin * sod, -cos * sod], 1)
    det = np.stack([-sin * (sdd - sod), cos * (sdd - sod)], 1)
    u = np.stack([cos * du, sin * du], 1)
    det = det + det_offset_u * u
    half = int(image_size) * voxel_size / 2
    return dict(image_size=int(image_size), det_cols=int(det_cols), sod=float(sod),
                sdd=float(sdd), voxel_size=float(voxel_size), det_pixel=du,
                angles=angles.astype(np.float32), det_offset_u=float(det_offset_u),
                vectors=np.concatenate([src, det, u], 1), window=(-half, half, -half, half))


def obliquity_weights(geometry):
    """1/cos(ray, detector normal) per detector pixel.

    Shape (rows, 1, cols) for a cone geometry and (1, 1, cols) for a fan
    geometry. For a circular orbit the detector moves rigidly with the
    source, so the weight is the same for every view.
    """
    vec = np.asarray(geometry["vectors"], np.float64)
    if vec.shape[1] == 6:
        src, det, u = vec[0, 0:2], vec[0, 2:4], vec[0, 4:6]
        normal = np.array([-u[1], u[0]]) / np.linalg.norm(u)
        c = np.arange(geometry["det_cols"]) - (geometry["det_cols"] - 1) / 2
        ray = det + c[:, None] * u - src
        cos = np.abs(ray @ normal) / np.linalg.norm(ray, axis=-1)
        return (1.0 / cos)[None, None, :].astype(np.float32)
    rows, cols = geometry["det_shape"]
    src, det, u, v = vec[0, 0:3], vec[0, 3:6], vec[0, 6:9], vec[0, 9:12]
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    r = np.arange(rows) - (rows - 1) / 2
    c = np.arange(cols) - (cols - 1) / 2
    pixels = det + r[:, None, None] * v + c[None, :, None] * u
    ray = pixels - src
    cos = np.abs(ray @ normal) / np.linalg.norm(ray, axis=-1)
    return (1.0 / cos)[:, None, :].astype(np.float32)


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, op):
        ctx.op = op
        return op._apply(volume, back=False)

    @staticmethod
    def backward(ctx, grad):
        return _BackProject.apply(grad, ctx.op), None


class _BackProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projection, op):
        ctx.op = op
        return op._apply(projection, back=True)

    @staticmethod
    def backward(ctx, grad):
        return _Project.apply(grad, ctx.op), None


class _ASTRA3D(LinearOperator):
    nonnegative_matrix = True
    """FP3D_CUDA / BP3D_CUDA on GPU-linked buffers for a ``*_vec`` geometry."""

    _projection_type = None

    def __init__(self, geometry, obliquity=False, device=None):
        try:
            import astra
        except ImportError as error:
            raise ImportError(f"{type(self).__name__} requires astra-toolbox "
                              "(conda install -c astra-toolbox astra-toolbox)") from error
        if not torch.cuda.is_available():
            raise RuntimeError(f"{type(self).__name__} requires a CUDA GPU")
        self.astra = astra
        self.device = (torch.device("cuda", torch.cuda.current_device()) if device is None
                       else torch.device(device))
        if self.device.type != "cuda":
            raise ValueError(f"{type(self).__name__} runs on CUDA only")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.geometry = geometry
        nz, ny, nx = geometry["volume_shape"]
        rows, cols = geometry["det_shape"]
        self.num_angles = len(geometry["angles"])
        self.angles = torch.as_tensor(geometry["angles"], dtype=torch.float32)
        self.domain_shape = (nz, ny, nx)
        self.range_shape = (rows, self.num_angles, cols)
        self._pg = astra.create_proj_geom(self._projection_type, rows, cols, geometry["vectors"])
        x0, x1, y0, y1, z0, z1 = geometry["window"]
        self._vg = astra.create_vol_geom(ny, nx, nz, x0, x1, y0, y1, z0, z1)
        with torch.cuda.device(self.device):
            self._volume = torch.zeros(self.domain_shape, dtype=torch.float32, device=self.device)
            self._projection = torch.zeros(self.range_shape, dtype=torch.float32, device=self.device)
        self._vid = astra.data3d.link("-vol", self._vg, astra.data3d.GPULink(self._volume.data_ptr(), nx, ny, nz, nx * 4))
        self._pid = astra.data3d.link("-sino", self._pg, astra.data3d.GPULink(self._projection.data_ptr(), cols, self.num_angles, rows, cols * 4))
        option = {"GPUindex": self.device.index}
        fp = astra.astra_dict("FP3D_CUDA")
        fp.update(VolumeDataId=self._vid, ProjectionDataId=self._pid, option=option)
        bp = astra.astra_dict("BP3D_CUDA")
        bp.update(ReconstructionDataId=self._vid, ProjectionDataId=self._pid, option=option)
        self._fp = astra.algorithm.create(fp)
        self._bp = astra.algorithm.create(bp)
        self._weight = (torch.from_numpy(obliquity_weights(geometry)).to(self.device)
                        if obliquity else None)

    def _run(self, algorithm):
        torch.cuda.synchronize(self.device)
        self.astra.algorithm.run(algorithm)
        torch.cuda.synchronize(self.device)

    def _astra_apply(self, value, back):
        """Batched FP (back=False) or weighted BP (back=True) without autograd."""
        in_shape, out_shape = ((self.range_shape, self.domain_shape) if back
                               else (self.domain_shape, self.range_shape))
        if tuple(value.shape[1:]) != in_shape:
            raise ValueError(f"expected (B, {in_shape}), got {tuple(value.shape)}")
        source, target = ((self._projection, self._volume) if back
                          else (self._volume, self._projection))
        out = torch.empty((value.shape[0], *out_shape), dtype=torch.float32, device=self.device)
        for i in range(value.shape[0]):
            item = value[i].detach().to(device=self.device, dtype=torch.float32)
            if back and self._weight is not None:
                item = item * self._weight
            # Public volumes use descending world Y, ASTRA 3D uses ascending Y.
            source.copy_(item if back else item.flip(-2))
            target.zero_()
            self._run(self._bp if back else self._fp)
            out[i].copy_(target.flip(-2) if back else target)
        return out.to(value.dtype) if value.is_floating_point() else out

    def _apply(self, value, back):
        from .cone_projector import cone_apply
        projection = dict(self._pg, ProjectionAngles=self.geometry['angles'])
        return cone_apply(value, self._vg, projection, not back)

    def forward(self, x):
        return _Project.apply(x, self)

    def adjoint(self, y):
        return _BackProject.apply(y, self)

    def close(self):
        """Release the ASTRA algorithms and linked data objects."""
        if getattr(self, "_fp", None) is not None:
            self.astra.algorithm.delete([self._fp, self._bp])
            self.astra.data3d.delete([self._vid, self._pid])
            self._fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ConeBeamASTRA(_ASTRA3D):
    """Matched midpoint ray projection and transpose with ASTRA analytic reconstruction."""

    _projection_type = "cone_vec"

    def __init__(self, geometry, obliquity=True, device=None):
        if geometry.get("kind", "cone") != "cone":
            raise ValueError("ConeBeamASTRA needs a circular_cone_geometry")
        super().__init__(geometry, obliquity=obliquity, device=device)

    def fdk(self, y, filter_type="ram-lak", short_scan=False, voxel_supersampling=1):
        """ASTRA FDK_CUDA reconstruction of a batch of projections."""
        if filter_type != "ram-lak":
            raise ValueError("FDK supports only ram-lak")
        from .astra_fdk_adapter import _voxel_sizes
        sizes = _voxel_sizes(self._vg)
        if not np.allclose(sizes, sizes[0], rtol=1e-6, atol=1e-7):
            raise ValueError("FDK requires isotropic voxels")
        from .astra_fdk_adapter import run_fdk_projection
        options = dict(GPUindex=self.device.index, ShortScan=bool(short_scan),
                       VoxelSuperSampling=int(voxel_supersampling))
        images = [run_fdk_projection(item, self._vg, self._pg, options) for item in y]
        return torch.stack(images).flip(-2).to(device=y.device, dtype=y.dtype)

    def pseudo_inverse(self, y, method="fdk", **kwargs):
        if method == "fdk":
            return self.fdk(y, **kwargs)
        from inv_framework.solvers.algebraic import landweber, sirt
        if method == "sirt":
            return sirt(self, y, **kwargs)
        if method == "landweber":
            return landweber(self, y, **kwargs)
        raise ValueError(f"Unknown pseudo_inverse method: {method!r}")



class ParallelBeamASTRA3D(_ASTRA3D):
    """Matched midpoint ray projection and transpose with ASTRA analytic reconstruction."""

    _projection_type = "parallel3d_vec"

    def __init__(self, geometry, device=None):
        if geometry.get("kind") != "parallel3d":
            raise ValueError("ParallelBeamASTRA3D needs a parallel_beam_3d_geometry")
        super().__init__(geometry, obliquity=False, device=device)

    def fbp(self, y, filter_type="ram-lak"):
        """Ram-Lak filtering and BP for an isotropic physical volume grid.

        Assumes angles uniformly covering [0, pi) or [0, 2 pi).
        """
        if filter_type != "ram-lak":
            raise ValueError("ParallelBeamASTRA3D.fbp implements only filter_type='ram-lak'")
        from .astra_fdk_adapter import _voxel_sizes
        sizes = _voxel_sizes(self._vg)
        if not np.allclose(sizes, sizes[0], rtol=1e-6, atol=1e-7):
            raise ValueError("ParallelBeamASTRA3D.fbp requires isotropic voxels")
        from inv_framework.solvers.analytic import _filter_sino
        filtered = _filter_sino(y.to(self.device, torch.float32))
        # The world-unit ramp filter carries 1/du; ASTRA's discrete BP3D carries
        # du/voxel^2 relative to the continuous back-projection (measured over
        # voxel and pitch sweeps), leaving 1/voxel^2.
        voxel = sizes[0]
        return self._astra_apply(filtered, True) * (math.pi / self.num_angles / voxel ** 2)

    def pseudo_inverse(self, y, method="fbp", **kwargs):
        if method == "fbp":
            return self.fbp(y, **kwargs)
        from inv_framework.solvers.algebraic import landweber, sirt
        if method == "sirt":
            return sirt(self, y, **kwargs)
        if method == "landweber":
            return landweber(self, y, **kwargs)
        raise ValueError(f"Unknown pseudo_inverse method: {method!r}")


class FanBeamASTRA2D(LinearOperator):
    """Circular fan CT with matched ray weights and native ASTRA analytic FBP."""

    nonnegative_matrix = True

    def __init__(self, geometry, obliquity=True, device=None):
        try:
            import astra
        except ImportError as error:
            raise ImportError("FanBeamASTRA2D requires astra-toolbox "
                              "(conda install -c astra-toolbox astra-toolbox)") from error
        if not torch.cuda.is_available():
            raise RuntimeError("FanBeamASTRA2D requires a CUDA GPU")
        self.astra = astra
        self.device = torch.device("cuda", torch.cuda.current_device() if device is None
                                   else torch.device(device).index or 0)
        self.geometry = geometry
        n, cols = geometry["image_size"], geometry["det_cols"]
        self.num_angles = len(geometry["angles"])
        self.angles = torch.as_tensor(geometry["angles"], dtype=torch.float32)
        self.image_size = n
        self.domain_shape = (1, n, n)
        self.range_shape = (1, self.num_angles, cols)
        self._pg = astra.create_proj_geom("fanflat_vec", cols, geometry["vectors"])
        x0, x1, y0, y1 = geometry["window"]
        self._vg = astra.create_vol_geom(n, n, x0, x1, y0, y1)
        self._volume = np.zeros((n, n), np.float32)
        self._projection = np.zeros((self.num_angles, cols), np.float32)
        self._vid = astra.data2d.link("-vol", self._vg, self._volume)
        self._pid = astra.data2d.link("-sino", self._pg, self._projection)
        option = {"GPUindex": self.device.index}
        fp = astra.astra_dict("FP_CUDA")
        fp.update(VolumeDataId=self._vid, ProjectionDataId=self._pid, option=option)
        bp = astra.astra_dict("BP_CUDA")
        bp.update(ReconstructionDataId=self._vid, ProjectionDataId=self._pid, option=option)
        self._fp = astra.algorithm.create(fp)
        self._bp = astra.algorithm.create(bp)
        self._weight = (torch.from_numpy(obliquity_weights(geometry)[0]) if obliquity else None)

    def _apply(self, value, back):
        from .cone_projector import cone_apply
        vectors = self.geometry['vectors']
        spatial = np.zeros((len(vectors), 12))
        spatial[:, :2] = vectors[:, :2]
        spatial[:, 3:5] = vectors[:, 2:4]
        spatial[:, 6:8] = vectors[:, 4:6]
        spatial[:, 11] = 1.
        n = self.image_size
        x0, x1, y0, y1 = self.geometry['window']
        volume = dict(GridSliceCount=1, GridRowCount=n, GridColCount=n,
                      option=dict(WindowMinX=x0, WindowMaxX=x1,
                                  WindowMinY=y0, WindowMaxY=y1,
                                  WindowMinZ=-.5, WindowMaxZ=.5))
        projection = dict(type='cone_vec', Vectors=spatial,
                          ProjectionAngles=self.geometry['angles'],
                          DetectorRowCount=1, DetectorColCount=self.range_shape[-1])
        return cone_apply(value, volume, projection, not back)

    def forward(self, x):
        return _Project.apply(x, self)

    def adjoint(self, y):
        return _BackProject.apply(y, self)

    def fbp(self, y, filter_type="ram-lak", short_scan=False):
        """Rebin a circular flat detector to its centered ASTRA FBP geometry."""
        if short_scan:
            raise ValueError("fan FBP short_scan is unsupported by this ASTRA backend")
        geometry = self.geometry
        expected = fan_beam_geometry(self.image_size, self.range_shape[-1],
            geometry['sod'], geometry['sdd'], voxel_size=geometry['voxel_size'],
            det_pixel=geometry['det_pixel'], angles=geometry['angles'],
            det_offset_u=geometry['det_offset_u'])
        if not np.allclose(geometry['vectors'], expected['vectors'], rtol=1e-6, atol=1e-4):
            raise ValueError("fan FBP requires the declared circular flat-detector orbit")
        astra = self.astra
        projection_geometry = astra.create_proj_geom('fanflat', geometry['det_pixel'],
            self.range_shape[-1], geometry['angles'], geometry['sod'],
            geometry['sdd']-geometry['sod'])
        x0, x1, y0, y1 = geometry['window']
        n = self.image_size
        dx, dy = (x1-x0)/n, (y1-y0)/n
        if not np.isclose(dx, dy, rtol=1e-6):
            raise ValueError("fan FBP requires isotropic pixels")
        # ASTRA FBP requires a grid centered on the circular rotation axis.
        canvas_size = math.ceil(2*max(abs(x0), abs(x1), abs(y0), abs(y1))/dx)
        half = canvas_size*dx/2
        volume_geometry = astra.create_vol_geom(canvas_size, canvas_size, -half, half, -half, half)
        xs = torch.arange(n, dtype=torch.float32)*dx + x0 + dx/2
        ys = y1 - (torch.arange(n, dtype=torch.float32)+.5)*dy
        gy, gx = torch.meshgrid(-ys/half, xs/half, indexing='ij')
        grid = torch.stack((gx, gy), -1)[None]
        projection_id = reconstruction_id = algorithm = None
        try:
            projection_id = astra.data2d.create('-sino', projection_geometry)
            reconstruction_id = astra.data2d.create('-vol', volume_geometry, 0.)
            config = astra.astra_dict('FBP_CUDA')
            config.update(ProjectionDataId=projection_id, ReconstructionDataId=reconstruction_id,
                          option=dict(GPUindex=self.device.index, FilterType=filter_type))
            algorithm = astra.algorithm.create(config)
            columns = np.arange(self.range_shape[-1])
            sample = columns-geometry['det_offset_u']
            images = []
            for item in y[:, 0].detach().float().cpu().numpy():
                centered = np.stack([np.interp(sample, columns, row, left=0., right=0.) for row in item])
                astra.data2d.store(projection_id, np.asarray(centered, dtype=np.float32))
                astra.data2d.store(reconstruction_id, 0.)
                astra.algorithm.run(algorithm)
                canvas = torch.from_numpy(astra.data2d.get(reconstruction_id).copy())[None, None]
                sampled = torch.nn.functional.grid_sample(canvas, grid, align_corners=False)
                images.append(sampled[0, 0])
            return torch.stack(images)[:, None].to(device=y.device, dtype=y.dtype)
        finally:
            if algorithm is not None:
                astra.algorithm.delete(algorithm)
            if reconstruction_id is not None:
                astra.data2d.delete(reconstruction_id)
            if projection_id is not None:
                astra.data2d.delete(projection_id)

    def pseudo_inverse(self, y, method="fbp", **kwargs):
        if method == "fbp":
            return self.fbp(y, **kwargs)
        from inv_framework.solvers.algebraic import landweber, sirt
        if method == "sirt":
            return sirt(self, y, **kwargs)
        if method == "landweber":
            return landweber(self, y, **kwargs)
        raise ValueError(f"Unknown pseudo_inverse method: {method!r}")

    def close(self):
        if getattr(self, "_fp", None) is not None:
            self.astra.algorithm.delete([self._fp, self._bp])
            self.astra.data2d.delete([self._vid, self._pid])
            self._fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
