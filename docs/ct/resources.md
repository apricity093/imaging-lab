# CT 请求、资源与独立安装

## 文件任务模板

将模板复制到工作目录，替换观测与模型资源。相对路径按任务文件目录解析，内存 Python 请求中的路径按调用工作目录解析。角度、探测器尺寸、距离、体素大小必须与实测数据一致。观测为物理域线积分，参考图仅放在 evaluation.reference，data_range 按实际评价协议填写。

| 用途 | 模板 | 布局与条件 |
| --- | --- | --- |
| 二维平行束 FBP | [原有模板](request_template.yaml) | (B,1,8,64)，CPU Torch |
| 截断 SIRT | [truncated_request](../../configs/ct/truncated_request.yaml) | (B,1,8,64)，显式中心32列掩膜 |
| 二维扇束 | [fan_2d_request](../../configs/ct/fan_2d_request.yaml) | (B,1,12,20)，16²，ASTRA CUDA |
| 三维平行束 | [parallel_3d_request](../../configs/ct/parallel_3d_request.yaml) | (B,8,12,20)，体积8×16×16，ASTRA CUDA |
| 三维锥束 FDK | [cone_3d_request](../../configs/ct/cone_3d_request.yaml) | (B,10,12,20)，体积8×16×16，等方体素，ASTRA CUDA |
| 像素域 DDS | [pixel_diffusion_request](../../configs/ct/pixel_diffusion_request.yaml) | (B,1,40,512)，512²，unet/ 与 scheduler/ |
| 潜空间 PSLD | [latent_diffusion_request](../../configs/ct/latent_diffusion_request.yaml) | 同上，另需 vqvae/ |
| NaSn pMF | [nasn_reconstruction](../../configs/ct/nasn_reconstruction.yaml) | (1,1,900,2048)，预处理、标定、权重及 Toolkit/MSVC |

深度请求保留 [docs 模板](deep_request_template.yaml)。缺失真实模型和观测时需先替换资源。pMF 的训练2048网格、相邻模型 JSON 和标定尺度须保持匹配。公开 pMF 任务1–20步，来源独立 benchmark 的32步协议另见 NaSn 手册。

```powershell
inv-run --config REQUEST.yaml --output OUTPUT
```

## 安装与定位

```powershell
python -m pip install .
# 按任务选用 diffusion、nasn、nasn-train extras
inv-download-weights --help
inv-download-weights lodoind-pixel --output D:/data/weights
inv-download-weights lodoind-latent --output D:/data/weights
inv-download-weights pmf --output D:/data/weights
inv-run --help
invct --help
inv-nasn-ct --help
inv-nasn-ct-train --help
```

wheel 与 sdist 含算法默认配置、上述 configs/ct 模板、NaSn 配置和清单、pMF 资源清单、四份 C++/CUDA 源文件与许可证。基础安装惰性加载可选依赖。ASTRA、Gridrec/NaSn 编译依赖、数据与模型权重按实际任务准备。

```python
from pathlib import Path
import configs
import inv_framework.applications.nasn_ct as nasn
config_root = Path(configs.__file__).parent
nasn_root = Path(nasn.__file__).parent
# Copy config_root / 'ct/fan_2d_request.yaml' to a task directory and fill inputs.
```

模型下载入口安装后可独立使用。来源 toy 数据、生成工具、算法卡片和手册保留在仓库，toy 的生成与预处理参数、来源许可见 [八类数据说明](../../examples/ct_data/README.md)。求解过程读取本地权重。DM4CT 与 pMF 发布效果仍需要真实匹配资源验收，当前小网络流程检查按其实际范围记录。

## 文档维护

在仓库根运行 `python -m tools.render_algorithm_docs`，索引取实际注册表。source_cards 保存完整来源卡片，supplements 保存同名经典方法的补充正文，两者在生成方法页时合并。编辑补充正文时更新对应资源，参数定义与默认配置更新后重新生成。共享方法页同时保留 MRI 复数、Hermitian 伴随与反卷积边界、PSF、噪声模型及领域默认值。CT 深度方法按 CT 注册，领域专属契约见对应手册。
