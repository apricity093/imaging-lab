# DPS（Diffusion Posterior Sampling）

在 DDPM 反向采样中注入测量残差梯度。

## 实现与执行约定

固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。实现位于 inv_framework/solvers/deep，公开 ID 为 dps。

适用于与预训练先验同网格（像素模型 512²、潜空间模型 512² 图像）的二维 CT，任意提供正投影与伴随的线性几何（平行束、扇束）；先验在 LoDoInd 工业 CT 切片上训练，跨数据域需要重新训练或微调。不支持三维体数据；所有结果保持模型域 [-1, 1] 的尺度约定。

Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023（arXiv:2209.14687）

CT 公共任务接收 float32 图像 (B,C,H,W)、观测 (B,C,角度,探测器列)，支持二维平行束与扇束。独立求解器保留原生类型及构造参数。模型域转换、资源与运行条件见 [深度先验](../ct/deep_priors.md)。

## 公共参数

| 参数 | 默认值 | 含义与约束 | 来源与推荐范围 |
| --- | --- | --- | --- |
| num_inference_steps | 1000 | DDPM 采样步数 {"type": "int", "nullable": false, "minimum": 1, "maximum": 1000, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "正整数，不超过训练步数 1000"} | DM4CT 原实验脚本（reverse_sample_*_ct.py，commit 49b3e59）； |
| scale | 1.0 | 测量引导步长（NormalizedPosteriorSampling） {"type": "float", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": false, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "正实数"} | 本仓库 toy 数据扫描（LoDoInd 40 角平行束，scale 0.3/1/3/10/30 → 21.6/22.2/16.1/10.8/9.0 dB）；DM4CT 原设置为 10；0.3–1；3 以上明显发散 |
| weights | None | Local pretrained model directory or pMF safetensors file with adjacent JSON {"type": "str", "nullable": true, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| seed | 0 | Seed for network initialization and sampling random streams {"type": "int", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| image_scale | 1.0 | Positive physical image scale in x_phys = image_scale*x_model + image_offset {"type": "float", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": false, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| image_offset | 0.0 | Physical image offset, included in observation conversion through A(offset*1) {"type": "float", "nullable": false, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |

YAML 默认值和参数含义来自实际注册信息，来源卡片保留推荐值依据。显式参数覆盖任务文件，任务文件覆盖算法配置。直接构造 DIP 默认 2000 次、EMA 关闭，推荐配置为 3000 次和 EMA 0.99。

## 独立调用与文件任务

```python
from inv_framework.ct.toy import toy_experiment
from inv_framework.runner import run_experiment
config = toy_experiment('examples/ct_data/toy_lodoind_512.npz', 'dps')
# Set config['params']['weights'], image_scale and image_offset for the selected prior.
result = run_experiment(config, output_dir='outputs/dps')
```

完整文件请求见 docs/ct/deep_request_template.yaml。使用 python -m inv_framework.runner --config REQUEST.yaml --output OUTPUT 执行，输出 result.json 和物理域 reconstruction.npy。上层调用使用相同 CT API 和声明的外层预算，无需 Agent SDK。

## 验证与资源限制

模型资源和运行条件见 [深度先验](../ct/deep_priors.md)。短预算执行只说明流程可用，不能代替推荐预算下的重建效果评价。

## 数学与方法差异

DIP/INR 在固定观测下优化网络参数，使 A fθ 与 y 匹配。扩散方法的去噪估计、梯度引导、伪逆投影与数据一致性方程，以及 pMF 速度积分和 LoRA 内层更新见 [方法公式](../ct/diffusion_methods.md)。各算法的来源参数依据与使用限制保留在下方卡片。


## 来源卡片记录

固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。以下测量、环境与评价口径为来源记录。当前公共输出、参数映射、离散化和限制见 CT 手册及本页公共参数。来源模型域输出映射为公共物理域后保存。

# DPS（Diffusion Posterior Sampling）


## 1. 算法身份

- **名称**：DPS（Diffusion Posterior Sampling）；注册名 `dps`
- **一句话简介**：在 DDPM 反向采样中注入测量残差梯度。
- **所属任务**：CT：二维平行束（同步辐射）、二维扇束（实验室锥束中心层）
- **方法类型**：学习类（使用预训练模型）
- **适用范围**：适用于与预训练先验同网格（像素模型 512²、潜空间模型 512² 图像）的二维 CT，任意提供正投影与伴随的线性几何（平行束、扇束）；先验在 LoDoInd 工业 CT 切片上训练，跨数据域需要重新训练或微调。不支持三维体数据；所有结果保持模型域 [-1, 1] 的尺度约定。
- **对应论文**：Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023（arXiv:2209.14687）
- **代码来源**：改编自 DM4CT（https://github.com/DM4CT/DM4CT ，commit 49b3e59，pipelines.py / condition_methods.py）；预训练权重 jiayangshi/lodoind_*_diffusion（MIT）

## 2. 输入输出

示例数据位于 `examples/ct_data/`，字段、单位与许可见 [数据说明](../../examples/ct_data/README.md)。每个数据包包含真值 `image`、投影 `sinogram`、角度 `theta`（弧度）、有测量的探测器列 `det_mask`与几何参数：

| 文件 | 几何 | 内容 | 预处理 |
| --- | --- | --- | --- |
| `toy_lodoind_512.npz` | 二维平行束（同步辐射） | LoDoInd 测试切片 2750（工业 CT），40 个角度（0–180°）稀疏角，无噪声（DM4CT 设置）。`image` 512×512，`sinogram` 40×512（角度×探测器） | DM4CT `preprocess_lodoind.py`（百分位截断、映射到 [-1, 1]、缩放到 512²）后取 `(x + 1) / 2` |
| `toy_lodoind_fan_512.npz` | 二维扇束（实验室锥束中心层，ASTRA） | 同一 LoDoInd 切片，实验室扇束 80 个角度（0–360°），SOD 1024、SDD 1536，576 个探测器，无噪声。`image` 512×512，`sinogram` 80×576（角度×探测器） | 同上；`sinogram` 由 ASTRA 扇束正投影生成 |

公共 runner 在 `--output` 目录生成物理域 `reconstruction.npy` 和 `result.json`，后者包含实际参数、耗时、指标、诊断及输出清单。二维输出包含 batch、channel 轴，三维包含 batch 轴。

**正常结果**（推荐默认参数、种子 0；CPU 行为服务器 CPU，GPU 行为 NVIDIA A100-SXM4-80GB）：

| 数据 | 输出尺寸 | PSNR (dB) | SSIM | ROI PSNR (dB) | 测量相对残差 | 耗时 (s) | 设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy_lodoind_512.npz` | 512×512 | 22.25 | 0.579 | — | 0.0118 | 485.0 | cuda |
| `toy_lodoind_fan_512.npz` | 512×512 | 23.14 | 0.557 | — | 0.0190 | 483.6 | cuda |

PSNR/SSIM 以数据包真值的极差为动态范围；截断（内部断层）数据另给中心视场圆内的 ROI 指标。表中数值及来源的 1 dB 容差属于来源记录，本项目的当前验证见 CT 验证文档。

说明：DM4CT 原设置 scale=10 在本仓库 LoDoInd toy 数据上发散（平行束 10.8 dB），默认值改为 1。归一化损失把恰好为 0 的测量视为未测量，扇束数据（探测器超出体积）不再被放大步长。

## 3. 运行方式

```bash
pip install -e ".[diffusion]"
conda install -c astra-toolbox astra-toolbox=2.4   # 三维平行束、扇束、锥束数据需要
python tools/download_deep_weights.py lodoind-pixel
python -m examples.run_deep_ct --algorithm dps --input examples/ct_data/toy_lodoind_512.npz --output outputs/dps/toy_lodoind_512 --weights weights/lodoind_pixel_diffusion --device cuda
python -m examples.run_deep_ct --algorithm dps --input examples/ct_data/toy_lodoind_fan_512.npz --output outputs/dps/toy_lodoind_fan_512 --weights weights/lodoind_pixel_diffusion --device cuda
```

- 设备：需要 CUDA GPU。三维平行束、扇束与锥束数据使用 ASTRA CUDA 投影器，需要 GPU 与 astra-toolbox；`tomopy_cuda` 数据首次运行会用 ninja 编译 CUDA 扩展（需 `CUDA_HOME`）。
- 测试环境：Python 3.10、PyTorch 2.7.0+cu118、diffusers 0.32.2、astra-toolbox 2.4.0。
- 权重：DM4CT 在 LoDoInd 上训练的像素域 UNet（`unet/`、`scheduler/`，455 MB，MIT，固定版本 jiayangshi/lodoind_pixel_diffusion@0f4a0dc），下载到 `weights/lodoind_pixel_diffusion/`。
- 权重与数据的约定：模型在 512² 网格、[-1, 1] 值域上训练；`examples.run_deep_ct` 按数据包的 `image_scale=0.5`、`image_offset=0.5` 把测量换到模型域（`(y − A·0.5)/0.5`），重建后再映射回 [0, 1]。网格必须与模型一致（512²），否则报错。
- 参数通过 `--params` 传入 JSON 对象，例如 `--params '{"seed": 0}'`。能力及默认值通过 `inv_framework.ct.api.list_capabilities()` 查询。

## 4. 参数

| 参数名 | 作用 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围（可选） |
| --- | --- | --- | --- | --- | --- |
| `num_inference_steps` | DDPM 采样步数 | `1000` | DM4CT 原实验脚本（reverse_sample_*_ct.py，commit 49b3e59） | 正整数，不超过训练步数 1000 |  |
| `scale` | 测量引导步长（NormalizedPosteriorSampling） | `1` | 本仓库 toy 数据扫描（LoDoInd 40 角平行束，scale 0.3/1/3/10/30 → 21.6/22.2/16.1/10.8/9.0 dB）；DM4CT 原设置为 10 | 正实数 | 0.3–1；3 以上明显发散 |
