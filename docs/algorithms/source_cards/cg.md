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

示例数据位于 `examples/ct_data/`，字段、单位与许可见 [数据说明](../../../examples/ct_data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

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
