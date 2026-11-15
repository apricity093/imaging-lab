# pMF-DDS（mean-flow 先验 + 每步 CG）

2048² pixel MeanFlow 先验的 DDS 采样，每步做 Tikhonov-CG 数据一致性。

## 实现与执行约定

固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。实现位于 inv_framework/solvers/deep，公开 ID 为 pmf_dds。

适用于 2048² 单通道二维 CT，包括内部断层（物体大于探测器视场的截断投影）与稀疏角扫描；先验在 NaSn 电池 FullCT 重建切片上训练。需要 CUDA GPU。网格必须为 2048²；跨材料/跨设备数据属于分布外，需配合 LoRA 适配或重新训练；不支持三维体数据。

Chung, Lee & Ye, DDS, ICLR 2024；Geng et al., Mean Flows for One-step Generative Modeling, 2025；Lu et al., One-step Latent-free Image Generation with Pixel Mean Flows, 2026（arXiv:2601.22158）

CT 公共任务接收 float32 图像 (B,C,H,W)、观测 (B,C,角度,探测器列)，支持二维平行束与扇束。独立求解器保留原生类型及构造参数。模型域转换、资源与运行条件见 [深度先验](../ct/deep_priors.md)。

## 公共参数

| 参数 | 默认值 | 含义与约束 | 来源与推荐范围 |
| --- | --- | --- | --- |
| num_inference_steps | 16 | mean-flow 采样步数 {"type": "int", "nullable": false, "minimum": 1, "maximum": 20, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "1–20 的整数；超过 20 步时 t<0.05，pMF 的 x0 估计不再精确"} | nasn-truncated-ct 发布配置（合成 benchmark 门槛 16 步）；16 或 20 |
| dc_lam | 5.0 | CG 中的 Tikhonov 权重 λ（模型域） {"type": "float", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": false, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "正实数；随测量尺度变化"} | nasn-truncated-ct 合成 benchmark；合成数据 5.0；真实 NaSn 数据 0.05 |
| dc_iter | 10 | 每步 CG 迭代次数 {"type": "int", "nullable": false, "minimum": 1, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "正整数"} | nasn-truncated-ct 发布配置；5–20 |
| eta | 0.0 | 每步注入新噪声的比例 {"type": "float", "nullable": false, "minimum": 0, "maximum": 1, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "[0, 1]"} | nasn-truncated-ct 发布配置（确定性采样）； |
| omega | 1.0 | pMF 的 CFG 系数 {"type": "float", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": false, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "正实数；无条件模型取 1"} | nasn-truncated-ct 发布配置； |
| weights | None | Local pretrained model directory or pMF safetensors file with adjacent JSON {"type": "str", "nullable": true, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| seed | 0 | Seed for network initialization and sampling random streams {"type": "int", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| image_scale | 1.0 | Positive physical image scale in x_phys = image_scale*x_model + image_offset {"type": "float", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": false, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| image_offset | 0.0 | Physical image offset, included in observation conversion through A(offset*1) {"type": "float", "nullable": false, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| model_config | None | Explicit local pMF JSON configuration {"type": "str", "nullable": true, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| t_start | 1.0 |  {"type": "float", "nullable": false, "minimum": 0, "maximum": 1, "inclusive_minimum": false, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| warm_start | False |  {"type": "bool", "nullable": false, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |

YAML 默认值和参数含义来自实际注册信息，来源卡片保留推荐值依据。显式参数覆盖任务文件，任务文件覆盖算法配置。直接构造 DIP 默认 2000 次、EMA 关闭，推荐配置为 3000 次和 EMA 0.99。

## 独立调用与文件任务

```python
from inv_framework.ct.toy import toy_experiment
from inv_framework.runner import run_experiment
config = toy_experiment('examples/ct_data/toy_nasn_pmf_2048.npz', 'pmf_dds')
# Set config['params']['weights'], image_scale and image_offset for the selected prior.
result = run_experiment(config, output_dir='outputs/pmf_dds')
```

完整文件请求见 docs/ct/deep_request_template.yaml。使用 python -m inv_framework.runner --config REQUEST.yaml --output OUTPUT 执行，输出 result.json 和物理域 reconstruction.npy。上层调用使用相同 CT API 和声明的外层预算，无需 Agent SDK。

## 验证与资源限制

模型资源和运行条件见 [深度先验](../ct/deep_priors.md)。短预算执行只说明流程可用，不能代替推荐预算下的重建效果评价。

## 数学与方法差异

DIP/INR 在固定观测下优化网络参数，使 A fθ 与 y 匹配。扩散方法的去噪估计、梯度引导、伪逆投影与数据一致性方程，以及 pMF 速度积分和 LoRA 内层更新见 [方法公式](../ct/diffusion_methods.md)。各算法的来源参数依据与使用限制保留在下方卡片。


## 来源卡片记录

固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。以下测量、环境与评价口径为来源记录。当前公共输出、参数映射、离散化和限制见 CT 手册及本页公共参数。来源模型域输出映射为公共物理域后保存。

# pMF-DDS（mean-flow 先验 + 每步 CG）


## 1. 算法身份

- **名称**：pMF-DDS（mean-flow 先验 + 每步 CG）；注册名 `pmf_dds`
- **一句话简介**：2048² pixel MeanFlow 先验的 DDS 采样，每步做 Tikhonov-CG 数据一致性。
- **所属任务**：CT：二维平行束（同步辐射）、二维扇束（实验室锥束中心层）
- **方法类型**：学习类（使用预训练模型）
- **适用范围**：适用于 2048² 单通道二维 CT，包括内部断层（物体大于探测器视场的截断投影）与稀疏角扫描；先验在 NaSn 电池 FullCT 重建切片上训练。需要 CUDA GPU。网格必须为 2048²；跨材料/跨设备数据属于分布外，需配合 LoRA 适配或重新训练；不支持三维体数据。
- **对应论文**：Chung, Lee & Ye, DDS, ICLR 2024；Geng et al., Mean Flows for One-step Generative Modeling, 2025；Lu et al., One-step Latent-free Image Generation with Pixel Mean Flows, 2026（arXiv:2601.22158）
- **代码来源**：迁移自 peterw634/nasn-truncated-ct（commit 35bb708，BSD-3-Clause）；pMF 模型保留上游 MIT 许可；权重 nasn-pmf-2048-epoch449-ema（model-v0.1.0，MIT）

## 2. 输入输出

示例数据位于 `examples/ct_data/`，字段、单位与许可见 [数据说明](../../examples/ct_data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

| 文件 | 几何 | 内容 | 预处理 |
| --- | --- | --- | --- |
| `toy_nasn_pmf_2048.npz` | 二维平行束内部断层（同步辐射，NaSn 几何） | pMF 无条件生成的电池切片（不含 NaSn 扫描数据），NaSn 合成 benchmark 几何：2048 个角度，仅中心 512/2048 列有测量（内部断层），无噪声。`image` 2048×2048，`sinogram` 2048×2048（角度×探测器） | `image = (pMF 样本 + 1) / 2`（float16 存储）；未测量列存为 0 |
| `toy_nasn_pmf_fan_2048.npz` | 二维扇束（实验室锥束中心层，ASTRA） | 同一 pMF 切片，实验室扇束 180 个角度（0–360°），SOD 4096、SDD 6144，2304 个探测器，全视场，无噪声。`image` 2048×2048，`sinogram` 180×2304（角度×探测器） | 同上；`sinogram` 由 ASTRA 扇束正投影生成 |

公共 runner 在 `--output` 目录生成物理域 `reconstruction.npy` 和 `result.json`，后者包含实际参数、耗时、指标、诊断及输出清单。二维输出包含 batch、channel 轴，三维包含 batch 轴。

**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 NVIDIA A100-SXM4-80GB）：

| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy_nasn_pmf_2048.npz` | 2048×2048 | 44.54 | 0.996 | 64.05 | 0.0002 | 65.2 | cuda |
| `toy_nasn_pmf_fan_2048.npz` | 2048×2048 | 66.24 | 1.000 | — | 0.0000 | 15.3 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。表中数值及来源的 1 dB 容差属于来源记录，本项目的当前验证见 CT 验证文档。

说明：toy 真值是该先验自身的无条件样本（分布内），因此近乎完美的恢复是预期结果，不代表真实数据质量；NaSn 真实合成验证集上 16 步的 ROI PSNR 为 36.6 dB（`https://github.com/peterw634/inv_framework/blob/dc181fb65afbf300a9935c3c770231c44ffc0c78/docs/nasn_ct.md`）。

## 3. 运行方式

```bash
pip install -e ".[diffusion]"
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python tools/download_deep_weights.py pmf
python -m examples.run_deep_ct --algorithm pmf_dds --input examples/ct_data/toy_nasn_pmf_2048.npz --output outputs/pmf_dds/toy_nasn_pmf_2048 --weights weights/nasn-pmf-2048-epoch449-ema.safetensors --device cuda
python -m examples.run_deep_ct --algorithm pmf_dds --input examples/ct_data/toy_nasn_pmf_fan_2048.npz --output outputs/pmf_dds/toy_nasn_pmf_fan_2048 --weights weights/nasn-pmf-2048-epoch449-ema.safetensors --device cuda
```

- 设备：需要 CUDA GPU。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：NaSn pMF-B/128 epoch-449 EMA 权重（502 MiB，MIT），来自 https://github.com/peterw634/nasn-truncated-ct/releases/tag/model-v0.1.0 ，脚本按清单校验 SHA-256；同名 `.json` 配置必须放在权重旁边。
- 权重与数据的约定：模型在 2048²、[-1, 1] 值域上训练；toy 包用 `x_phys = 0.5·x_model + 0.5`（与 NaSn 合成 benchmark 相同），此时推荐 λ=5.0。真实 NaSn 测量使用 `physical_image × scale` 的标定且不做中心化，对应 λ=0.05（见 `inv-nasn-ct reconstruct`）。网格必须为 2048²。
- 参数通过 `--params` 传入 JSON 对象，例如 `--params '{"seed": 0}'`。能力及默认值通过 `inv_framework.ct.api.list_capabilities()` 查询。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `num_inference_steps` | mean-flow 采样步数 | `16` | nasn-truncated-ct 发布配置（合成 benchmark 门槛 16 步） | 1–20 的整数；超过 20 步时 t<0.05，pMF 的 x0 估计不再精确 | 16 或 20 |
| `dc_lam` | CG 中的 Tikhonov 权重 λ（模型域） | `5` | nasn-truncated-ct 合成 benchmark | 正实数；随测量尺度变化 | 合成数据 5.0；真实 NaSn 数据 0.05 |
| `dc_iter` | 每步 CG 迭代次数 | `10` | nasn-truncated-ct 发布配置 | 正整数 | 5–20 |
| `eta` | 每步注入新噪声的比例 | `0` | nasn-truncated-ct 发布配置（确定性采样） | [0, 1] |  |
| `omega` | pMF 的 CFG 系数 | `1` | nasn-truncated-ct 发布配置 | 正实数；无条件模型取 1 |  |
