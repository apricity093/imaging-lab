"""Algorithm registry: identity, recommended defaults and a uniform builder.

Each entry describes one reconstruction algorithm in the form used by
``algorithms/<name>/README.md`` and ``examples/run.py``. ``defaults(name)``
returns the recommended parameters; ``build(name, ...)`` returns a callable
``solve(measurement) -> image`` working in the prior's model units (identical
to physical units for non-learned methods).

Defaults are the values each algorithm card documents, with their source.
Heavy dependencies (diffusers, pMF, CUDA extensions) are imported only when an
algorithm that needs them is built.
"""
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class Param:
    name: str
    default: Any
    help: str
    source: str
    constraint: str
    recommended: str = ""


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    title: str
    summary: str
    learned: bool
    weights: Optional[str]          # None, 'dm4ct-pixel', 'dm4ct-latent' or 'pmf'
    toy_data: Tuple[str, ...]       # synchrotron parallel-beam packs first, lab geometry last
    device: str                     # minimum device for the parallel-beam pack: 'cpu' or 'cuda'
    params: Tuple[Param, ...]
    builder: Callable = field(repr=False, compare=False, default=None)
    # Card text (algorithms/<name>/README.md): supported tasks, scope, paper, code origin.
    task: str = ""
    scope: str = ""
    paper: str = ""
    code: str = ""


def _p(*args, **kwargs):
    return Param(*args, **kwargs)


DM4CT_SOURCE = "DM4CT 原实验脚本（reverse_sample_*_ct.py，commit 49b3e59）"
SIRT_ITER = _p("sirt_iterations", 100, "伪逆/投影所用 SIRT 的迭代次数", "DM4CT 参考算子（SIRTProjectionOperator）",
               "正整数", "50–200")


# --------------------------------------------------------------------------- builders

def _build_dip(operator, params, **_):
    from inv_framework.solvers.deep.dip import DIPSolver
    solver = DIPSolver(num_iterations=params["num_iterations"], lr=params["lr"],
                       input_channels=params["input_channels"], ema_decay=params["ema_decay"])
    return _bound_solver(solver, operator)


def _build_inr(operator, params, **_):
    from inv_framework.solvers.deep.inr import INRSolver
    solver = INRSolver(num_iterations=params["num_iterations"], lr=params["lr"],
                       hidden_features=params["hidden_features"],
                       hidden_layers=params["hidden_layers"])
    return _bound_solver(solver, operator)


_DM4CT_CLASSES = {
    "mcg": "MCGSolver", "pgdm": "PGDMSolver", "ddim_dds": "DDIMDDSSolver",
    "dmplug": "DMPlugSolver", "reddiff": "REDDiffSolver", "hybridreg": "HybridRegSolver",
    "psld": "PSLDSolver", "resample": "ReSampleSolver", "diffstategrad": "DiffStateGradSolver",
}
_DDPM = {"dps", "mcg", "psld"}
_LATENT = {"psld", "resample", "diffstategrad"}


def _build_dm4ct(operator, params, *, name, weights, device, seed, **_):
    import safetensors.torch
    from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel, VQModel
    from inv_framework.solvers.deep.models.latent_codec import VQCodec
    from inv_framework.core.sirt_projection import SIRTProjectionOperator
    from inv_framework.solvers.deep import diffusion as ds
    if weights is None:
        raise ValueError(f"{name} needs pretrained DM4CT weights (--weights DIR)")
    params = dict(params)
    net = UNet2DModel.from_pretrained(weights, subfolder="unet", local_files_only=True)
    kind = DDPMScheduler if name in _DDPM else DDIMScheduler
    schedule = kind.from_pretrained(weights, subfolder="scheduler", local_files_only=True)
    if schedule.config.prediction_type != "epsilon":
        raise ValueError(f"{name} requires an epsilon-prediction scheduler")
    codec = None
    if name in _LATENT:
        vq = VQModel.from_pretrained(weights, subfolder="vqvae", local_files_only=True)
        codec = VQCodec(vq.to(device).eval())
    else:
        size = net.config.sample_size
        grid = (size, size) if isinstance(size, int) else tuple(size)
        if grid != tuple(operator.domain_shape[-2:]):
            raise ValueError(f"{name}: model grid {grid} differs from the data grid "
                             f"{tuple(operator.domain_shape[-2:])}")
    if not name in _LATENT and net.config.in_channels != operator.domain_shape[0]:
        raise ValueError(f"{name}: model channels differ from operator domain")
    net = net.to(device).eval()
    A = SIRTProjectionOperator(operator, iterations=params.pop("sirt_iterations", 100))
    if name == "dps":
        scale = params.pop("scale")
        solver = ds.DPSSolver(ds.DiffusersUNetWrapper(net), ds.DiffusersScheduleAdapter(schedule),
                              conditioning=ds.NormalizedPosteriorSampling(A, scale=scale), **params)
    else:
        solver = getattr(ds, _DM4CT_CLASSES[name])(net, schedule, codec=codec, **params)
    generator_device = "cpu" if name in _LATENT else device

    def solve(y):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        return solver.solve(y, A, generator=generator)
    solve.solver = solver
    return solve


def _load_prior(weights, device, omega, config_path=None):
    from inv_framework.solvers.deep.models.weights import load_pmf
    from inv_framework.solvers.deep.diffusion import PMFVelocityPrior
    if weights is None:
        raise ValueError("pMF methods need the released weights (--weights FILE.safetensors)")
    return PMFVelocityPrior(load_pmf(weights, config_path=config_path, device=device), omega=omega)


def _noise(operator, device, seed, batch):
    from inv_framework.solvers.deep.random import BatchGenerator
    return BatchGenerator(device, [seed] * batch).randn((batch, *operator.domain_shape))


def _build_pmf_dds(operator, params, *, weights, device, seed, **_):
    from inv_framework.solvers.deep.diffusion import DDSSolver
    prior = _load_prior(weights, device, params["omega"], params.get("model_config"))
    if (prior.img_channels, prior.img_size, prior.img_size) != tuple(operator.domain_shape):
        raise ValueError("pMF model grid/channels differ from operator domain")
    solver = DDSSolver(prior, num_inference_steps=params["num_inference_steps"],
                       eta=params["eta"], dc_lam=params["dc_lam"], dc_iter=params["dc_iter"],
                       t_start=params.get("t_start", 1.0))
    return _bound_solver(solver, operator, seed=seed, device=device,
                         warm_start=params.get("warm_start", False))


def _build_pmf_lora_dds(operator, params, *, weights, device, seed, **_):
    from inv_framework.solvers.deep.diffusion import LoRAAdaptiveDDSSolver, adapt_pmfdit
    prior = _load_prior(weights, device, params["omega"], params.get("model_config"))
    if (prior.img_channels, prior.img_size, prior.img_size) != tuple(operator.domain_shape):
        raise ValueError("pMF model grid/channels differ from operator domain")
    groups = adapt_pmfdit(prior.net, r=params["lora_rank"], verbose=False)
    adapt = [p for group in groups for p in group]
    lr = params["adapt_lr"]
    solver = LoRAAdaptiveDDSSolver(
        prior, adapt_params=adapt, adapt_optim_factory=lambda p: torch.optim.Adam(p, lr=lr),
        adapt_K=params["adapt_K"], num_inference_steps=params["num_inference_steps"],
        eta=params["eta"], dc_lam=params["dc_lam"], dc_iter=params["dc_iter"])
    return _bound_solver(solver, operator, seed=seed, device=device)


# --------------------------------------------------------------------------- registry

_TOY_2D = ("examples/ct_data/toy_parallel_ct_128.npz", "examples/ct_data/toy_lodoind_fan_512.npz")
_TOY_LODOIND = ("examples/ct_data/toy_lodoind_512.npz", "examples/ct_data/toy_lodoind_fan_512.npz")
_TOY_PMF = ("examples/ct_data/toy_nasn_pmf_2048.npz", "examples/ct_data/toy_nasn_pmf_fan_2048.npz")

_SPECS = [
    AlgorithmSpec(
        "dip", "DIP（Deep Image Prior）", "用随机初始化 CNN 的结构先验，逐例拟合测量。", False, None,
        _TOY_2D, "cpu", (
            _p("num_iterations", 3000, "Adam 迭代次数", "本仓库 toy 数据上的经验值（配合 ema_decay）", "正整数",
               "2000–4000；有噪声时迭代过多会过拟合噪声"),
            _p("lr", 1e-3, "Adam 学习率", "本框架默认值；Ulyanov 等 2018 使用 1e-2–1e-3", "正实数",
               "1e-3；1e-2 在 2500 次后开始拟合噪声"),
            _p("input_channels", 32, "固定随机输入 z 的通道数", "本框架默认值（SkipUNet）", "正整数"),
            _p("ema_decay", 0.99, "输出取迭代结果的指数滑动平均", "DIP 官方代码 exp_weight=0.99",
               "None 或 [0, 1)", "None 时返回最后一次迭代，可能落在 Adam 尖峰上"),
        ), _build_dip),
    AlgorithmSpec(
        "inr", "INR（SIREN 隐式神经表示）", "用坐标 MLP 表示图像，逐例拟合测量。", False, None,
        _TOY_2D, "cpu", (
            _p("num_iterations", 2000, "Adam 迭代次数", "本框架默认值", "正整数", "1000–5000"),
            _p("lr", 1e-4, "Adam 学习率", "本框架默认值；SIREN 论文常用 1e-4", "正实数"),
            _p("hidden_features", 256, "隐藏层宽度", "本框架默认值", "正整数"),
            _p("hidden_layers", 4, "隐藏层数", "本框架默认值", "正整数"),
        ), _build_inr),
    AlgorithmSpec(
        "dps", "DPS（Diffusion Posterior Sampling）", "在 DDPM 反向采样中注入测量残差梯度。", True,
        "dm4ct-pixel", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 1000, "DDPM 采样步数", DM4CT_SOURCE, "正整数，不超过训练步数 1000"),
            _p("scale", 1.0, "测量引导步长（NormalizedPosteriorSampling）",
               "本仓库 toy 数据扫描（LoDoInd 40 角平行束，scale 0.3/1/3/10/30 → 21.6/22.2/16.1/10.8/9.0 dB）；"
               "DM4CT 原设置为 10", "正实数", "0.3–1；3 以上明显发散"),
        ), _build_dm4ct),
    AlgorithmSpec(
        "mcg", "MCG（Manifold-Constrained Gradient）", "梯度引导后再投影到测量约束。", True,
        "dm4ct-pixel", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 1000, "DDPM 采样步数", DM4CT_SOURCE, "正整数，不超过 1000"),
            _p("scale", 2.0, "流形约束梯度步长", DM4CT_SOURCE, "正实数"),
            SIRT_ITER,
        ), _build_dm4ct),
    AlgorithmSpec(
        "pgdm", "ΠGDM（Pseudoinverse-Guided Diffusion）", "用伪逆构造引导方向的 DDIM 采样。", True,
        "dm4ct-pixel", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 100, "DDIM 采样步数", DM4CT_SOURCE, "正整数"),
            _p("scale", 0.3, "伪逆引导系数", DM4CT_SOURCE, "正实数"),
            SIRT_ITER,
        ), _build_dm4ct),
    AlgorithmSpec(
        "ddim_dds", "DDS（Decomposed Diffusion Sampler，DDIM 版）", "DDIM 采样中对 Tweedie 估计做 CG 数据一致性。", True,
        "dm4ct-pixel", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 100, "DDIM 采样步数", DM4CT_SOURCE, "正整数"),
            _p("eta", 0.85, "DDIM 随机性", DM4CT_SOURCE, "[0, 1]"),
            _p("cg_inner", 5, "每步 CG 迭代次数", DM4CT_SOURCE, "正整数", "3–10"),
            _p("gamma", 0.0, "CG 方程中的 Tikhonov 项", DM4CT_SOURCE, "非负实数"),
        ), _build_dm4ct),
    AlgorithmSpec(
        "dmplug", "DMPlug", "把整条短采样轨迹视为可微映射，优化初始噪声。", True,
        "dm4ct-pixel", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 3, "每次前向的 DDIM 步数", DM4CT_SOURCE, "正整数；显存随步数线性增长", "1–5"),
            _p("epochs", 1000, "初始噪声的优化次数", DM4CT_SOURCE, "正整数"),
            _p("lr", 0.005, "Adam 学习率", DM4CT_SOURCE, "正实数"),
        ), _build_dm4ct),
    AlgorithmSpec(
        "reddiff", "RED-diff", "以扩散去噪器为正则项，用 Adam 直接优化图像。", True,
        "dm4ct-pixel", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 200, "噪声等级数（优化步数）", DM4CT_SOURCE, "正整数"),
            _p("sigma", 1e-4, "测量噪声标准差", DM4CT_SOURCE, "非负实数"),
            _p("loss_measurement_weight", 0.5, "测量损失权重", DM4CT_SOURCE, "非负实数"),
            _p("loss_noise_weight", 10000.0, "扩散正则权重", DM4CT_SOURCE, "非负实数"),
            _p("lr", 0.099, "Adam 学习率", DM4CT_SOURCE, "正实数"),
            SIRT_ITER,
        ), _build_dm4ct),
    AlgorithmSpec(
        "hybridreg", "HybridReg", "RED-diff 的相关噪声 + 混合正则变体。", True,
        "dm4ct-pixel", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 200, "噪声等级数（优化步数）", DM4CT_SOURCE, "正整数"),
            _p("sigma", 1e-4, "测量噪声标准差", DM4CT_SOURCE, "非负实数"),
            _p("loss_measurement_weight", 0.5, "测量损失权重", DM4CT_SOURCE, "非负实数"),
            _p("loss_noise_weight", 10000.0, "扩散正则权重", DM4CT_SOURCE, "非负实数"),
            _p("lr", 0.099, "Adam 学习率", DM4CT_SOURCE, "正实数"),
            _p("beta", 0.9999, "相关噪声混合系数", DM4CT_SOURCE, "[0, 1)"),
            SIRT_ITER,
        ), _build_dm4ct),
    AlgorithmSpec(
        "psld", "PSLD（潜空间后验采样）", "在 VQ 潜空间做 DPS 式引导并加潜变量一致性项。", True,
        "dm4ct-latent", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 1000, "DDPM 采样步数", DM4CT_SOURCE, "正整数，不超过 1000"),
            _p("gamma", 0.8, "解码测量误差的引导权重", DM4CT_SOURCE, "非负实数"),
            _p("omega", 0.2, "潜变量一致性（gluing）权重", DM4CT_SOURCE, "非负实数"),
            _p("eta", 0.0, "DDPM/DDIM 随机性", "本框架默认值", "[0, 1]"),
            SIRT_ITER,
        ), _build_dm4ct),
    AlgorithmSpec(
        "resample", "ReSample", "潜空间 DDIM 中做硬数据一致性优化并随机重采样。", True,
        "dm4ct-latent", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 200, "DDIM 采样步数", DM4CT_SOURCE, "正整数"),
            _p("pixel_lr", 1e-4, "像素域一致性优化学习率", DM4CT_SOURCE, "正实数"),
            _p("latent_lr", 0.1, "潜空间一致性优化学习率", DM4CT_SOURCE, "正实数"),
            _p("inter_timesteps", 5, "两次硬一致性之间的步数间隔", "ReSample 原实现默认值", "正整数"),
            _p("pixel_maxiters", 2000, "像素域优化最大迭代", "ReSample 原实现默认值", "正整数"),
            _p("latent_maxiters", 500, "潜空间优化最大迭代", "ReSample 原实现默认值", "正整数"),
        ), _build_dm4ct),
    AlgorithmSpec(
        "diffstategrad", "DiffStateGrad", "在 ReSample 的一致性优化中把梯度投影到扩散状态的低秩子空间。", True,
        "dm4ct-latent", _TOY_LODOIND, "cuda", (
            _p("num_inference_steps", 200, "DDIM 采样步数", DM4CT_SOURCE, "正整数"),
            _p("pixel_lr", 1e-4, "像素域一致性优化学习率", DM4CT_SOURCE, "正实数"),
            _p("latent_lr", 0.1, "潜空间一致性优化学习率", DM4CT_SOURCE, "正实数"),
            _p("var_cutoff", 0.99, "SVD 累计方差截断（决定子空间秩）", DM4CT_SOURCE, "(0, 1]"),
            _p("inter_timesteps", 5, "两次硬一致性之间的步数间隔", "ReSample 原实现默认值", "正整数"),
            _p("period", 1, "重新计算 SVD 的周期", "DiffStateGrad 原实现默认值", "正整数"),
            _p("pixel_maxiters", 2000, "像素域优化最大迭代", "DiffStateGrad 原实现默认值", "正整数"),
            _p("latent_maxiters", 500, "潜空间优化最大迭代", "DiffStateGrad 原实现默认值", "正整数"),
        ), _build_dm4ct),
    AlgorithmSpec(
        "pmf_dds", "pMF-DDS（mean-flow 先验 + 每步 CG）", "2048² pixel MeanFlow 先验的 DDS 采样，每步做 Tikhonov-CG 数据一致性。", True,
        "pmf", _TOY_PMF, "cuda", (
            _p("num_inference_steps", 16, "mean-flow 采样步数", "nasn-truncated-ct 发布配置（合成 benchmark 门槛 16 步）",
               "1–20 的整数；超过 20 步时 t<0.05，pMF 的 x0 估计不再精确", "16 或 20"),
            _p("dc_lam", 5.0, "CG 中的 Tikhonov 权重 λ（模型域）", "nasn-truncated-ct 合成 benchmark",
               "正实数；随测量尺度变化", "合成数据 5.0；真实 NaSn 数据 0.05"),
            _p("dc_iter", 10, "每步 CG 迭代次数", "nasn-truncated-ct 发布配置", "正整数", "5–20"),
            _p("eta", 0.0, "每步注入新噪声的比例", "nasn-truncated-ct 发布配置（确定性采样）", "[0, 1]"),
            _p("omega", 1.0, "pMF 的 CFG 系数", "nasn-truncated-ct 发布配置", "正实数；无条件模型取 1"),
        ), _build_pmf_dds),
    AlgorithmSpec(
        "pmf_lora_dds", "pMF-LoRA-DDS（在环 LoRA 自适应）", "pMF-DDS 的每个采样步先用测量残差更新 LoRA，再做 DDS 校正，用于分布外数据。", True,
        "pmf", _TOY_PMF, "cuda", (
            _p("num_inference_steps", 16, "mean-flow 采样步数", "同 pMF-DDS", "1–20 的整数"),
            _p("dc_lam", 5.0, "CG 中的 Tikhonov 权重 λ（模型域）", "同 pMF-DDS 合成 benchmark", "正实数"),
            _p("dc_iter", 10, "每步 CG 迭代次数", "LoRAAdaptiveDDSSolver 默认值", "正整数"),
            _p("eta", 0.0, "每步注入新噪声的比例", "同 pMF-DDS", "[0, 1]"),
            _p("omega", 1.0, "pMF 的 CFG 系数", "同 pMF-DDS", "正实数"),
            _p("lora_rank", 8, "LoRA 秩", "examples/diffusion_ct_compare.py 默认值", "正整数", "4–16"),
            _p("adapt_lr", 1e-4, "LoRA 的 Adam 学习率", "examples/diffusion_ct_compare.py 默认值", "正实数"),
            _p("adapt_K", 4, "每个采样步的 LoRA 更新次数", "LoRAAdaptiveDDSSolver 默认值", "正整数", "1–8"),
        ), _build_pmf_lora_dds),
]

_ALL_GEOMETRIES = "CT：二维与三维平行束（同步辐射）、二维扇束与三维锥束（实验室）"
_2D = "CT：二维平行束（同步辐射）、二维扇束（实验室锥束中心层）"
_INFR = "本仓库编写（inv_framework）"
_DM4CT_CODE = ("改编自 DM4CT（https://github.com/DM4CT/DM4CT ，commit 49b3e59，pipelines.py / "
               "condition_methods.py）；预训练权重 jiayangshi/lodoind_*_diffusion（MIT）")
_DM4CT_SCOPE = ("适用于与预训练先验同网格（像素模型 512²、潜空间模型 512² 图像）的二维 CT，"
                "任意提供正投影与伴随的线性几何（平行束、扇束）；先验在 LoDoInd 工业 CT 切片上训练，"
                "跨数据域需要重新训练或微调。不支持三维体数据；所有结果保持模型域 [-1, 1] 的尺度约定。")
_PMF_CODE = ("迁移自 peterw634/nasn-truncated-ct（commit 35bb708，BSD-3-Clause）；pMF 模型保留上游 MIT 许可；"
             "权重 nasn-pmf-2048-epoch449-ema（model-v0.1.0，MIT）")
_INFO = {
    "dip": (_2D,
            "适用于没有训练数据的二维重建：随机初始化的 CNN 逐例拟合测量，网络结构本身充当先验。"
            "不使用预训练模型（非学习类）。需要 GPU 才有实用速度；结果随随机种子波动，有噪声时需控制迭代次数；不支持三维。",
            "Ulyanov, Vedaldi & Lempitsky, Deep Image Prior, CVPR 2018", _INFR),
    "inr": (_2D,
            "适用于没有训练数据的二维重建：用 SIREN 坐标网络表示图像并逐例拟合测量，平滑区域效果好。"
            "不使用预训练模型（非学习类）。每个样本单独拟合一个网络；对学习率敏感；不支持三维。",
            "Sitzmann et al., Implicit Neural Representations with Periodic Activation Functions, NeurIPS 2020",
            _INFR),
    "dps": (_2D, _DM4CT_SCOPE,
            "Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023（arXiv:2209.14687）",
            _DM4CT_CODE),
    "mcg": (_2D, _DM4CT_SCOPE,
            "Chung et al., Improving Diffusion Models for Inverse Problems using Manifold Constraints, NeurIPS 2022（arXiv:2206.00941）",
            _DM4CT_CODE),
    "pgdm": (_2D, _DM4CT_SCOPE,
             "Song, Vahdat, Mardani & Kautz, Pseudoinverse-Guided Diffusion Models for Inverse Problems, ICLR 2023",
             _DM4CT_CODE),
    "ddim_dds": (_2D, _DM4CT_SCOPE,
                 "Chung, Lee & Ye, Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems, ICLR 2024（arXiv:2303.05754）",
                 _DM4CT_CODE),
    "dmplug": (_2D, _DM4CT_SCOPE + " 每次优化都对整条采样轨迹反传，显存随步数增长。",
               "Wang et al., DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models, NeurIPS 2024（arXiv:2405.16749）",
               _DM4CT_CODE),
    "reddiff": (_2D, _DM4CT_SCOPE,
                "Mardani et al., A Variational Perspective on Solving Inverse Problems with Diffusion Models, ICLR 2024（arXiv:2305.04391）",
                _DM4CT_CODE),
    "hybridreg": (_2D, _DM4CT_SCOPE,
                  "Dou et al., Hybrid Regularization Improves Diffusion-based Inverse Problem Solving, ICLR 2025",
                  _DM4CT_CODE),
    "psld": (_2D, _DM4CT_SCOPE,
             "Rout et al., Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models, NeurIPS 2023（arXiv:2307.00619）",
             _DM4CT_CODE),
    "resample": (_2D, _DM4CT_SCOPE,
                 "Song et al., Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency, ICLR 2024（arXiv:2307.08123）",
                 _DM4CT_CODE),
    "diffstategrad": (_2D, _DM4CT_SCOPE,
                      "Zirvi, Tolooshams & Anandkumar, Diffusion State-Guided Projected Gradient for Inverse Problems, ICLR 2025（arXiv:2410.03463）",
                      _DM4CT_CODE),
    "pmf_dds": (_2D,
                "适用于 2048² 单通道二维 CT，包括内部断层（物体大于探测器视场的截断投影）与稀疏角扫描；"
                "先验在 NaSn 电池 FullCT 重建切片上训练。需要 CUDA GPU。网格必须为 2048²；"
                "跨材料/跨设备数据属于分布外，需配合 LoRA 适配或重新训练；不支持三维体数据。",
                "Chung, Lee & Ye, DDS, ICLR 2024；Geng et al., Mean Flows for One-step Generative Modeling, 2025；"
                "Lu et al., One-step Latent-free Image Generation with Pixel Mean Flows, 2026（arXiv:2601.22158）",
                _PMF_CODE),
    "pmf_lora_dds": (_2D,
                     "在 pMF-DDS 的基础上，每个采样步先以测量残差（经可微 CG）更新注入 pmfDiT 的 LoRA，"
                     "用于测量与先验分布不一致的数据。需要 CUDA GPU 与更多显存（对 2048² DiT 反传）；"
                     "网格必须为 2048²；不支持三维体数据。",
                     "Chung & Ye, Deep Diffusion Image Prior for Efficient OOD Adaptation in 3D Inverse Problems, "
                     "ECCV 2024（arXiv:2407.10641）；Hu et al., LoRA, 2021",
                     f"{_INFR}（solvers/diffusion/dds_lora.py、lora.py，配方移植自 indus_CT/NaSn lora_pmfdit）；"
                     "pMF 部分同 pMF-DDS"),
}
_SPECS = [replace(spec, task=_INFO[spec.name][0], scope=_INFO[spec.name][1],
                  paper=_INFO[spec.name][2], code=_INFO[spec.name][3]) for spec in _SPECS]

ALGORITHMS: Dict[str, AlgorithmSpec] = {spec.name: spec for spec in _SPECS}


def defaults(name: str) -> Dict[str, Any]:
    return {p.name: p.default for p in ALGORITHMS[name].params}


def resolve_params(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Recommended defaults updated with ``overrides``; unknown names raise."""
    params = defaults(name)
    unknown = set(overrides or {}) - set(params)
    if unknown:
        raise KeyError(f"{name} has no parameter(s) {sorted(unknown)}; choose from {sorted(params)}")
    params.update(overrides or {})
    return params


def build(name: str, operator, params: Dict[str, Any], *, weights=None,
          device="cpu", seed=0) -> Callable[[torch.Tensor], torch.Tensor]:
    spec = ALGORITHMS[name]
    return spec.builder(operator, params, name=name, weights=weights,
                        device=str(device), seed=seed)


def _bound_solver(solver, operator, seed=None, device=None, warm_start=False):
    def solve(y):
        options = {"x_init": operator.pseudo_inverse(y)} if warm_start else {}
        if seed is not None:
            options["initial_noise"] = _noise(operator, device, seed, y.shape[0])
        return solver.solve(y, operator, **options)
    solve.solver = solver
    return solve

def _build_budgeted(operator, params, *, weights, device, seed, **_):
    import safetensors.torch
    from diffusers import DDPMScheduler, UNet2DModel
    from inv_framework.solvers.deep.diffusion import BudgetedDPSSolver, DiffusersUNetWrapper, DiffusersScheduleAdapter
    from inv_framework.solvers.deep.diffusion.budget_dps import BudgetSegmentAction
    from inv_framework.core.budget import SubsettableForwardOperator
    if not isinstance(operator, SubsettableForwardOperator):
        raise ValueError("budgeted_dps requires source-subset physics")
    if not params.get("schedule_actions"):
        raise ValueError("budgeted_dps requires an explicit schedule_actions array")
    if not weights:
        raise ValueError("budgeted_dps requires local pixel diffusion weights")
    net = UNet2DModel.from_pretrained(weights, subfolder="unet", local_files_only=True).to(device).eval()
    size = net.config.sample_size
    grid = (size, size) if isinstance(size, int) else tuple(size)
    if (net.config.in_channels, *grid) != tuple(operator.domain_shape):
        raise ValueError("budgeted_dps model grid/channels differ from operator domain")
    schedule = DDPMScheduler.from_pretrained(weights, subfolder="scheduler", local_files_only=True)
    options = dict(params)
    actions = [BudgetSegmentAction(tuple(indices)) for indices in options.pop("schedule_actions")]
    if any(i >= operator.num_sources for a in actions for i in a.subset):
        raise ValueError("schedule_actions source index exceeds operator.num_sources")
    solver = BudgetedDPSSolver(DiffusersUNetWrapper(net), actions,
        schedule=DiffusersScheduleAdapter(schedule), **options)
    def solve(y):
        return solver.solve(y, operator, generator=torch.Generator(device=device).manual_seed(seed))
    solve.solver = solver
    return solve

ALGORITHMS["budgeted_dps"] = AlgorithmSpec(
    "budgeted_dps", "Budgeted DPS", "Explicit source-subset DPS conditioning.",
    True, "dm4ct-pixel", _TOY_LODOIND, "cuda", (
        Param("num_inference_steps", 200, "Reverse sampling steps", "source constructor", "positive integer"),
        Param("scale", 1.0, "Residual gradient scale", "source constructor", "positive real"),
        Param("eta", 0.0, "Scheduler stochasticity", "source constructor", "[0,1]"),
        Param("schedule_actions", None, "Source indices per segment", "explicit caller schedule", "nonempty array of nonempty nonnegative integer arrays"),
        Param("key_indices", None, "Optional segment start steps", "source constructor", "null or nonnegative integer array"),
    ), _build_budgeted, task="2D CT with source-subset physics",
    scope="Code and interface migration. Explicit schedules supplied by the caller.",
    code="peterw634/inv_framework dc181fb65afbf300a9935c3c770231c44ffc0c78")
