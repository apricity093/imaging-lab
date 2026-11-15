# SIRT

## 一、算法身份

- **名称**：SIRT。
- **一句话简介**：用行列归一化的同时残差更新执行代数重建。
- **所属任务**：CT。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：支持二维平行束、二维扇束、三维平行束及三维锥束 CT，可设置图像上下界。
- **代码来源**：[apricity093/imaging-lab](https://github.com/apricity093/imaging-lab)，包版本`0.1.1`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/solvers/algebraic.py](../../inv_framework/solvers/algebraic.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| CT / `sirt` | Simultaneous iterative reconstruction. Diagonally normalized simultaneous residual iteration for Ax=y; optional box projection, not an arbitrary likelihood solver. |

## 二、输入输出

### 最小示例数据

tests/data/cases/parallel_2d__ellipses_sparse_gaussian/arrays.h5：truth/x=(1,1,64,64)，measurement/y_observed=(1,1,32,64)，投影轴为批次、通道、角度、探测器。case.json 随附弧度角及归一化距离；原图固定乘0.02。投影加入1% RMS高斯噪声，不作负对数。

示例配置或生成脚本：[configs/algorithms/sirt.yaml](../../configs/algorithms/sirt.yaml)。

**正常输出**：二维(1,1,64,64)重建；reconstruction.pt 保存tensor bundle，comparison.png可直接查看，metrics.json与diagnostics.json保存指标、参数和停止状态。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- CT：[ct输入契约](../ct/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CPU。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

### 最小示例命令

```bash
invct run sirt --case parallel_2d__ellipses_sparse_gaussian --config configs/algorithms/sirt.yaml --out outputs/algorithm_examples/sirt --device cpu
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### CT：`sirt`

默认值来源：[本仓库配置](../../configs/algorithms/sirt.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Maximum outer iterations.；iterations | 25 | `configs/algorithms/sirt.yaml` | int；≥1 | [12, 25, 50] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/algorithms/sirt.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/algorithms/sirt.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |

保真项模式：`fixed`；可执行项：`['l2']`。Only registered updates; WLS requires public weights, Gaussian NLL public sigma, transmission NLL public counts/I0 and nonnegative constraints.

### 耦合条件

上下界须有序，迭代预算是上限，不保证恰好执行到上限。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。

## 工业 CT 数学与入口映射

更新为 xₖ₊₁ = P盒(xₖ + C A* R(y−Axₖ))，R 与 C 分别由行和、列和归一化得到。公共 SIRT 同时支持工业几何，dtype 跟随观测。来源推荐 200 次及非负约束在 industrial_baselines.yaml 保留。

几何、来源与限制见 [工业 CT](../ct/industrial.md)，两侧调用见 [Harness](../ct/harness.md)。


## 来源卡片记录

固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。以下测量、环境与评价口径为来源记录。当前公共输出、参数映射、离散化和限制见 CT 手册及本页公共参数。来源模型域输出映射为公共物理域后保存。

# SIRT（联合迭代重建）


## 1. 算法身份

- **名称**：SIRT（联合迭代重建）；注册名 `sirt`
- **一句话简介**：按行/列和归一化的迭代代数重建。
- **所属任务**：CT：二维与三维平行束（同步辐射）、二维扇束与三维锥束（实验室）
- **方法类型**：非学习类（不使用预训练模型）
- **适用范围**：适用于任意提供正投影与伴随的线性 CT 几何，对稀疏角与噪声比 FBP 稳健；可加非负/上界约束。无需训练。收敛慢于 CG，迭代过多会放大噪声；不含先验，强欠采样下仍有伪影。
- **对应论文**：Gilbert, Iterative methods for the three-dimensional reconstruction of an object from projections, J. Theor. Biol. 1972
- **代码来源**：本仓库编写（inv_framework）

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
| `toy_parallel_ct_128.npz` | 128×128 | 24.16 | 0.674 | — | 0.0205 | 5.8 | cpu |
| `toy_parallel_ct_3d.npz` | 64×128×128 | 29.00 | 0.919 | — | 0.0142 | 3.2 | cuda |
| `toy_cone_ct_3d.npz` | 64×128×128 | 25.73 | 0.914 | — | 0.0151 | 1.8 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。表中数值及来源的 1 dB 容差属于来源记录，本项目的当前验证见 CT 验证文档。

## 3. 运行方式

```bash
pip install -e .
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python -m examples.run_industrial_ct --algorithm sirt --input examples/ct_data/toy_parallel_ct_128.npz --output outputs/sirt/toy_parallel_ct_128 --device cpu
python -m examples.run_industrial_ct --algorithm sirt --input examples/ct_data/toy_parallel_ct_3d.npz --output outputs/sirt/toy_parallel_ct_3d --device cuda
python -m examples.run_industrial_ct --algorithm sirt --input examples/ct_data/toy_cone_ct_3d.npz --output outputs/sirt/toy_cone_ct_3d --device cuda
```

- 设备：二维平行束示例可在 CPU 上运行，GPU 更快。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：不需要。
- 参数通过 `--params` 传入 JSON 对象，例如 `--params '{"num_iterations": 20}'`。能力及默认值通过 `inv_framework.ct.api.list_capabilities()` 查询。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `num_iterations` | 迭代次数 | `200` | 本仓库 toy 数据上的经验值（50→400 次 PSNR 20.9→24.9 dB，200 次后收益递减） | 正整数 | 100–400；无非负约束时迭代过多会放大噪声 |
| `min_value` | 每次迭代后的下限截断 | `0` | 经验值：衰减系数非负 | None 或实数 |  |
| `max_value` | 每次迭代后的上限截断 | `None` | 不截断 | None 或大于 min_value 的实数 |  |
