# 共轭梯度 CG

## 一、算法身份

- **名称**：共轭梯度 CG。
- **一句话简介**：求解 Hermitian 正定正规系统。
- **所属任务**：CT、MRI、反卷积。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：cg_sense 是 SENSE 入口；CT tikhonov 是采用 CG 的二次正则入口。Tikhonov 为目标项，不另计算法。独立 cg 要求正权重恒等正则。
- **代码来源**：[apricity093/imaging-lab](https://github.com/apricity093/imaging-lab)，包版本`0.1.1`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/core/krylov.py](../../inv_framework/core/krylov.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| CT / `cg` | Independent Hermitian positive definite normal-system CG. Independent Hermitian CG: 0.5 sum Q\|Ax-y\|^2 + 0.5 strength \|\|x\|\|^2; requires strength>0. |
| CT / `tikhonov` | Quadratic Tikhonov reconstruction. 0.5 sum W\|Ax-y\|^2 + 0.5 reg_strength \|\|Kx\|\|^2, K=I or gradient; CG normal system, unconstrained. |
| MRI / `cg` | Native complex 2D Cartesian SENSE 0.5 sum Q \|Ax-y\|^2 + 0.5 strength \|\|Lx\|\|^2; unconstrained |
| MRI / `cg_sense` | Cartesian or non-Cartesian; solves weighted complex normal equations 0.5 sum W\|PFSx-y\|^2 + 0.5 lam \|\|x\|\|^2, complex Hermitian CG. |
| 反卷积 / `cg` | Real scalar 2D known PSF; periodic or zero convolution boundary 0.5 sum Q \|Ax-y\|^2 + 0.5 strength \|\|Lx\|\|^2; unconstrained |

共享入口支持二维平行束CT、二维Cartesian SENSE及已知PSF二维反卷积。专用入口的维度、边界和正则约定保留各自语义；MRI使用原生复数和Hermitian伴随。共同线性基线只接收二次目标。

## 二、输入输出

### 最小示例数据

脚本生成固定4×4复数对象；单线圈maps、mask为1，中心正交FFT得到 kspace (1,1,4,4)。恒等正则权重0.08，无裁剪或强度拟合。

示例配置或生成脚本：[examples/quadratic_quickstart.py](../../examples/quadratic_quickstart.py)。

**正常输出**：(1,4,4) complex128 reconstruction.npy 和 result.json。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- CT：[ct输入契约](../ct/data_format.md)，说明数组轴、几何/采样、单位及预处理。
- MRI：[mri输入契约](../mri/data_format.md)，说明数组轴、几何/采样、单位及预处理。
- 反卷积：[deconvolution输入契约](../deconvolution/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CPU。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

### 最小示例命令

```bash
python examples/quadratic_quickstart.py --domain mri --method cg --output outputs/algorithm_examples/cg
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

脚本的`--domain`可选ct、mri、deconvolution。线性CT示例读取64×64椭圆catalog，组合CT示例生成2×2匹配投影；MRI保留非平凡相位，反卷积使用已知小PSF。完整数据起点见[组合示例](../../examples/composition/README.md)。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### CT：`cg`

默认值来源：[本仓库配置](../../configs/algorithms/cg.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Maximum outer iterations.；iterations | 200 | `configs/algorithms/cg.yaml` | int；≥1 | [100, 200, 400] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/algorithms/cg.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/algorithms/cg.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| data_fidelity | Optimization data term; public sigma, weights or counts required by selected term；enum | l2 | `configs/algorithms/cg.yaml` | str；l2 / gaussian_nll / wls | ['l2', 'gaussian_nll', 'wls'] |
| weight_source | Origin of public WLS weights; provided, known variance or matched post-log count approximation；enum | None | `configs/algorithms/cg.yaml` | str；可为null；known_variance / provided / post_log_poisson / post_log_poisson_gaussian | ['known_variance', 'provided', 'post_log_poisson', 'post_log_poisson_gaussian'] |
| regularizer | Prior selector; ADMM tv_haar is a joint objective with separate weights；enum | tikhonov | `configs/algorithms/cg.yaml` | str；tikhonov | ['tikhonov'] |
| strength | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.01 | `configs/algorithms/cg.yaml` | float；>0.0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| boundary | 正则差分边界；物理卷积边界由model单独声明；dimensionless | neumann | `configs/algorithms/cg.yaml` | str；neumann / periodic | ['neumann', 'periodic'] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/algorithms/cg.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| lam_rel | Relative identity weight; lambda=lam_rel times estimated squared operator norm.；dimensionless | None | `configs/algorithms/cg.yaml` | float；可为null；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Only registered updates; WLS requires public weights, Gaussian NLL public sigma, transmission NLL public counts/I0 and nonnegative constraints.

### CT：`tikhonov`

默认值来源：[本仓库配置](../../configs/algorithms/tikhonov.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Maximum outer iterations.；iterations | 100 | `configs/algorithms/tikhonov.yaml` | int；≥1 | [50, 100, 200] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/algorithms/tikhonov.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/algorithms/tikhonov.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| data_fidelity | Optimization data term; public sigma, weights or counts required by selected term；enum | l2 | `configs/algorithms/tikhonov.yaml` | str；l2 / gaussian_nll / wls | ['l2', 'gaussian_nll', 'wls'] |
| weight_source | Origin of public WLS weights; provided, known variance or matched post-log count approximation；enum | None | `configs/algorithms/tikhonov.yaml` | str；可为null；known_variance / provided / post_log_poisson / post_log_poisson_gaussian | ['known_variance', 'provided', 'post_log_poisson', 'post_log_poisson_gaussian'] |
| reg_strength | Regularizer weight; alias strength on composable CT solvers；objective/regularizer | 1e-07 | `configs/algorithms/tikhonov.yaml` | float；≥0.0 | [1e-09, 1e-08, 1e-07, 1e-06, 9.999999999999999e-06, 9.999999999999999e-05] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-07 | `configs/algorithms/tikhonov.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| regularizer | Prior selector; ADMM tv_haar is a joint objective with separate weights；enum | tikhonov | `configs/algorithms/tikhonov.yaml` | str；tikhonov / h1 | ['tikhonov', 'h1'] |
| strength | Regularizer weight on the stated unnormalized sum；objective/regularizer | None | `configs/algorithms/tikhonov.yaml` | float；可为null；≥0.0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Only registered updates; WLS requires public weights, Gaussian NLL public sigma, transmission NLL public counts/I0 and nonnegative constraints.

### MRI：`cg`

默认值来源：[本仓库配置](../../configs/mri/algorithms/cg.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Iteration budget; actual stop may be earlier；iterations | 200 | `configs/mri/algorithms/cg.yaml` | int；≥1 | [100, 200, 400] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/mri/algorithms/cg.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/mri/algorithms/cg.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| regularizer | Prior selector; ADMM tv_haar is a joint objective with separate weights；enum | tikhonov | `configs/mri/algorithms/cg.yaml` | str；tikhonov | ['tikhonov'] |
| strength | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.01 | `configs/mri/algorithms/cg.yaml` | float；>0.0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/mri/algorithms/cg.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| boundary | 正则差分边界；物理卷积边界由model单独声明；dimensionless | neumann | `configs/mri/algorithms/cg.yaml` | str；neumann / periodic | ['neumann', 'periodic'] |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Quadratic 0.5 sum W |Ax-y|^2; real inverse-variance weights, identity if absent. WLS requires explicit weights. Shared MRI: 2D Cartesian SENSE; deconvolution: known PSF, gain=1, background=0.

### MRI：`cg_sense`

默认值来源：[本仓库配置](../../configs/mri/algorithms/cg_sense.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| lam | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.0 | `configs/mri/algorithms/cg_sense.yaml` | float；≥0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| max_iter | Iteration budget; actual stop may be earlier；iterations | 100 | `configs/mri/algorithms/cg_sense.yaml` | int；≥1 | [50, 100, 200] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/mri/algorithms/cg_sense.yaml` | float；≥0 | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Complex weighted quadratic only; Gaussian uses supplied inverse-variance data_weights or identity; WLS requires explicit weights. Calibration, interpolation and learned inference have no selectable reconstruction data term.

### 反卷积：`cg`

默认值来源：[本仓库配置](../../configs/deconvolution/algorithms/cg.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Iteration budget; actual stop may be earlier；iterations | 200 | `configs/deconvolution/algorithms/cg.yaml` | int；≥1 | [100, 200, 400] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/deconvolution/algorithms/cg.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/deconvolution/algorithms/cg.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| regularizer | Prior selector; ADMM tv_haar is a joint objective with separate weights；enum | tikhonov | `configs/deconvolution/algorithms/cg.yaml` | str；tikhonov | ['tikhonov'] |
| strength | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.01 | `configs/deconvolution/algorithms/cg.yaml` | float；>0.0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/deconvolution/algorithms/cg.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| boundary | 正则差分边界；物理卷积边界由model单独声明；dimensionless | neumann | `configs/deconvolution/algorithms/cg.yaml` | str；neumann / periodic | ['neumann', 'periodic'] |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Quadratic 0.5 sum W |Ax-y|^2; real inverse-variance weights, identity if absent. WLS requires explicit weights. Shared MRI: 2D Cartesian SENSE; deconvolution: known PSF, gain=1, background=0.

### 耦合条件

权重对应未按像素数平均的目标。Haar层数须适配尺寸；rho为正且与正则权重独立；步长和CG精度依据实际算子与系统。MRI复数变量不接受实数盒约束，具体组合按入口条件执行。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

## 工业 CT 数学与入口映射

来源 cg_tikhonov 映射为 cg、regularizer=tikhonov。lam_rel 换算 λ=lam_rel×估计的 ||A||²，求解 (A*WA+λI)x=A*Wy。max_iter 映射 num_iterations，tol 映射 tolerance。来源推荐 max_iter=20、lam_rel=0.001、tol=1e-6。公共停止按本项目正规残差规约，来源停止与质量记录按来源条件解释。带先验中心的深度 CG 使用右端 A*Wy+λx_prior，可微校正由 solvers/deep/cg.py 提供。

几何、来源与限制见 [工业 CT](../ct/industrial.md)，两侧调用见 [Harness](../ct/harness.md)。


## 来源卡片记录

固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。以下测量、环境与评价口径为来源记录。当前公共输出、参数映射、离散化和限制见 CT 手册及本页公共参数。来源模型域输出映射为公共物理域后保存。

# CG-Tikhonov


## 1. 算法身份

- **名称**：CG-Tikhonov；注册名 `cg_tikhonov`
- **一句话简介**：用共轭梯度求解 (AᵀA+λI)x = Aᵀy 的 Tikhonov 正则重建。
- **所属任务**：CT：二维与三维平行束（同步辐射）、二维扇束与三维锥束（实验室）
- **方法类型**：非学习类（不使用预训练模型）
- **适用范围**：适用于任意线性 CT 几何，用 CG 快速求解 L2 正则最小二乘；同一 CG 也是 DDS 类方法的数据一致性步。无需训练。没有非负约束，L2 正则会抹平边缘；稀疏角下质量低于 SIRT。
- **对应论文**：Hestenes & Stiefel, Methods of conjugate gradients for solving linear systems, 1952；Tikhonov 正则化
- **代码来源**：本仓库编写（inv_framework）；CG 移植自 indus_CT/NaSn cg_consistency

## 2. 输入输出

示例数据位于 `examples/ct_data/`，字段、单位与许可见 [数据说明](../../examples/ct_data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

| 文件 | 几何 | 内容 | 预处理 |
| --- | --- | --- | --- |
| `toy_parallel_ct_128.npz` | 二维平行束（同步辐射） | 改进 Shepp-Logan 模体，30 个角度（0–180°）稀疏角，加性高斯噪声 σ=0.5。`image` 128×128，`sinogram` 30×128（角度×探测器） | 无；`sinogram` 已是线积分（负对数后）投影 |
| `toy_parallel_ct_3d.npz` | 三维平行束（同步辐射体数据，ASTRA） | 与锥束包相同的合成圆柱电池，同步辐射式三维平行束 90 个角度（0–180°），每个探测器行对应一个切片，高斯噪声 σ=0.5%·max。`image` 64×128×128，`sinogram` 64×90×128（探测器行×角度×探测器列） | 无；`sinogram` 为线积分，ASTRA 顺序（探测器行, 角度, 探测器列）；探测器像素 = 体素 |
| `toy_cone_ct_3d.npz` | 三维锥束（实验室，ASTRA） | 合成圆柱电池（钢壳、卷绕层、极柱、端盖、夹杂），圆轨道锥束 120 个角度（0–360°），SOD 400、SDD 600（放大 1.5 倍），高斯噪声 σ=0.5%·max。`image` 64×128×128，`sinogram` 80×120×144（探测器行×角度×探测器列） | 无；`sinogram` 为线积分，ASTRA 顺序（探测器行, 角度, 探测器列）；长度以体素为单位 |

公共 runner 在 `--output` 目录生成物理域 `reconstruction.npy` 和 `result.json`，后者包含实际参数、耗时、指标、诊断及输出清单。二维输出包含 batch、channel 轴，三维包含 batch 轴。

**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 NVIDIA A100-SXM4-80GB）：

| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy_parallel_ct_128.npz` | 128×128 | 20.25 | 0.375 | — | 0.0096 | 1.7 | cpu |
| `toy_parallel_ct_3d.npz` | 64×128×128 | 27.55 | 0.751 | — | 0.0099 | 1.0 | cuda |
| `toy_cone_ct_3d.npz` | 64×128×128 | 22.62 | 0.726 | — | 0.0146 | 1.4 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。表中数值及来源的 1 dB 容差属于来源记录，本项目的当前验证见 CT 验证文档。

## 3. 运行方式

```bash
pip install -e .
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python -m examples.run_industrial_ct --algorithm cg_tikhonov --input examples/ct_data/toy_parallel_ct_128.npz --output outputs/cg_tikhonov/toy_parallel_ct_128 --device cpu
python -m examples.run_industrial_ct --algorithm cg_tikhonov --input examples/ct_data/toy_parallel_ct_3d.npz --output outputs/cg_tikhonov/toy_parallel_ct_3d --device cuda
python -m examples.run_industrial_ct --algorithm cg_tikhonov --input examples/ct_data/toy_cone_ct_3d.npz --output outputs/cg_tikhonov/toy_cone_ct_3d --device cuda
```

- 设备：二维平行束示例可在 CPU 上运行，GPU 更快。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：不需要。
- 参数通过 `--params` 传入 JSON 对象，例如 `--params '{"max_iter": 20, "lam_rel": 0.001}'`。能力及默认值通过 `inv_framework.ct.api.list_capabilities()` 查询。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `lam_rel` | 相对正则权重，λ = lam_rel·||A||² | `0.001` | 本仓库 toy 数据上的经验值 | 非负实数 | 1e-4–1e-2；噪声越大取越大 |
| `max_iter` | CG 最大迭代次数 | `20` | 本仓库 toy 数据上的经验值（50 次开始拟合噪声） | 正整数 | 10–50 |
| `tol` | 相对残差提前停止阈值 | `1e-6` | 本仓库经验值 | 正实数 |  |
