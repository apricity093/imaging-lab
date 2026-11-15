# 工业 CT 与来源整合契约

## 固定版本

| 代码 | 固定提交 | 用途 |
| --- | --- | --- |
| imaging-lab | 7db81d213fe4f5c7b97c62114b8e5c3c5f63a0e8 | 整合前基线 |
| peterw634/inv_framework，yuan_cx/inv_framework_indus_ct_cards | dc181fb65afbf300a9935c3c770231c44ffc0c78 | 数值实现、卡片和数据来源 |
| inverse-agent-harness | a0e672e11cf29ca5b79f9e98f7f3a6acf94089df | 计划指定的调用方契约基准 |

来源仓库为 https://github.com/peterw634/inv_framework 。整合保留本项目领域 API、runner、ExperimentResult、result.json 和 reconstruction.npy 契约。调用顺序为能力发现、参数准备、领域校验、run_experiment、结果消费。Harness 配套交付、固定基准与补丁使用见 [Harness](harness.md)。

## 功能与参数映射

| 来源 | 本项目入口 | 参数及行为 |
| --- | --- | --- |
| classical.fbp | solvers.analytic.fbp、公共 fbp | 保留唯一 Ram-Lak 实现、补零和角度缩放；扇束和三维平行束调用几何专用 fbp |
| 来源 fbp 卡片的锥束分派 | 公共 fdk | toy_experiment 将锥束 fbp 显式改为 fdk |
| classical.sirt | solvers.algebraic.sirt、公共 sirt | num_iterations、x_init、min_value、max_value；保持观测 dtype |
| classical.landweber | solvers.algebraic.landweber、公共 landweber | step_size=None 使用确定性 12 次幂迭代估计的 0.9/估计值；显式步长保持原定义 |
| registry.cg_tikhonov | 公共 cg，regularizer=tikhonov | max_iter→num_iterations，tol→tolerance，lam_rel→strength=lam_rel×估计的平方范数；实际权重和估计值进入 effective_params |
| solvers.cg 的带先验校正 | solvers/deep/cg.py | 保留 (A*WA+λI)x=A*Wy+λx_prior 及可微路径要求；独立 cg 使用零中心目标 |
| operators.ct.astra_cuda | ct.backends.astra_cuda | fan_2d、parallel_3d、cone_3d；ASTRA 解析重建与共享匹配射线积分 |
| operators.ct.nasn_cuda、radon_cuda | ct.backends.nasn_cuda、radon_cuda | backend=nasn，CUDA JIT 惰性编译，Gridrec 作为低层伪逆 |
| 截断投影 | ct.industrial.DetectorMaskedOperator | A=T R，A*=R*T，detector_mask 显式给出 |
| 其他经典、MRI、反卷积 | 原有入口 | 沿用现有配置和契约 |
| DIP、INR、DPS、扩散、pMF、LoRA、Budgeted DPS | solvers/deep 与 CT 公共任务 | 来源 solvers、models、registry 与算法卡片作为迁入基准 |
| NaSn 应用 | applications/nasn_ct | 来源 applications/nasn_ct 的预处理、训练、EMA 导出及重建 |

Landweber 的加权或带二次正则任务继续使用已有增广算子的 Schur 上界，自动步长为 1/上界。幂迭代结果为估计值。旧实验应显式填写原步长 0.001。CG 的 lam_rel 与绝对 strength 同时传入时，相对权重优先，实际 strength 在结果中报告。lam_rel=0 对应半正定无正则系统，可能出现曲率或精度停止。

## 几何与观测

二维布局为图像 (B,C,H,W)、投影 (B,C,角度,探测器列)。扇束限定 C=1、方形图像，探测器宽度独立。三维布局为体积 (B,Z,Y,X)、投影 (B,探测器行,角度,探测器列)，支持非立方体体积。

所有几何声明 type、domain_shape、range_shape、angles_rad。工业 ASTRA 几何额外声明 voxel_size、det_pixel、det_offset_u。扇束 det_pixel 为标量，三维为 [dv,du]。扇束和锥束声明 source_origin_distance 与 origin_detector_distance，来源 sdd 映射为两者之和。vectors 与 window 可覆盖规则矢量和体积边界。长度采用同一物理单位，偏移 det_offset_u 采用探测器像素。旧规则锥束字段 detector_spacing_row/col 等继续可用。

backend=astra 与 device=cuda 用于工业扇束和三维任务。FDK 要求各向同性体素，非立方体体积可用。FBP/FDK 仅公开当前 Ram-Lak 路径，扇束短扫描未提供。二维平行束 torch 后端采用现有单位像素、等宽探测器约定。NaSn 后端支持 center，单通道方形图像和等宽探测器。

来源 sinogram 已是线积分，加载时保持物理域。detector_mask 为探测器列布尔数组，未测列同时作用于正向和伴随。零值不会自动产生掩膜。valid_measurement_mask 继续表达原有评价和保真权重语义。图像仅在 evaluation.reference 中参与重建后评价。image_scale、image_offset 为模型域资源契约，经典算法不转换数据尺度。

## 数据与模型资源映射

来源八类 NPZ 保存在 examples/ct_data，文件名、字段和许可说明保持对应。toy_experiment 将 image 放入独立评价，将 sinogram、theta、det_mask 和物理几何映射到公共请求。目录中的 README 保留来源说明，本地生成与预处理命令见数据说明。

| 资源 | 本地输入约定 | 实施位置 |
| --- | --- | --- |
| 二维平行束、截断、三维平行束、锥束、LoDoInd 平行/扇束、pMF 截断/扇束 | examples/ct_data/*.npz | 公共 toy 请求映射 |
| DM4CT 像素域 UNet 与 scheduler | 保留下载后的模型目录和相邻配置 | 本地资源及 harness 暂存 |
| DM4CT 潜域 UNet、scheduler、VQ codec | 分别保留目录结构与预测类型 | 本地资源及 harness 暂存 |
| pMF checkpoint 与模型配置 | 匹配分辨率、通道、velocity 预测和物理尺度 | 本地资源及 harness 暂存 |
| NaSn 原始扫描、FullCT、EMA 权重 | 来源清单及用户可访问路径 | NaSn 独立应用 |
| CUDA、ASTRA、Gridrec | 可选运行依赖；CUDA 源码随安装包提供 | 当前数值后端 |

权重通过 inv-download-weights 的固定资源声明获取，求解期间不下载。来源算法注册表 inv_framework/registry.py、algorithms/<id>/README.md、examples/data/README.md 与 THIRD_PARTY_NOTICES.md 均以本页固定提交解释。模型、数据和代码许可各自保留，LoDoInd 为 CC BY-NC 4.0。

## Python 调用

```python
from inv_framework.ct.toy import toy_experiment
from inv_framework.runner import run_experiment

config = toy_experiment('examples/ct_data/toy_truncated_ct_128.npz',
                        'cg_tikhonov', params={'max_iter': 20, 'lam_rel': 0.001})
result = run_experiment(config, output_dir='outputs/industrial_ct')
```

文件任务与 CLI 使用同一 model.geometry、model.measurement、inputs.observation 和 params 字段。NPY/NPZ 文件引用需显式提供批次和通道轴，原始来源 NPZ 使用 toy_experiment 补齐轴。上层 harness 需原样暂存几何、掩膜、后端和设备，并将质量参考留在独立评价阶段。

## 数值实现与限制

当前 ASTRA 2.2 的来源 FP/BP 加权对在小型可达几何上未满足项目离散伴随要求。迭代重建使用 cone_projector 的共享三线性中点积分权重及转置，扇束为单层矢量几何，三维平行束为平行射线矢量几何。解析 FBP/FDK 保留 ASTRA 路径。迭代离散化因此与来源 ASTRA 原生投影存在差别，不能据此声称来源数据逐值复现。该路径优先建立数学契约，大体积吞吐性能尚未标定。CUDA 计算实际精度为 float32。


模型资源固定版本如下。

- jiayangshi/lodoind_pixel_diffusion 使用 0f4a0dcf5768abec78fdbc01b95a83194175c0f1。
- jiayangshi/lodoind_latent_diffusion 使用 6f35a9b0198a3a7e6eec64b69112745b3a24a295。
- pMF 使用来源 manifests/model-v0.1.0.json 中 nasn-pmf-2048-epoch449-ema 资源及相邻配置。


来源 toy 数据的独立命令如下。

```powershell
python -m examples.run_industrial_ct --input examples/ct_data/toy_parallel_ct_128.npz --algorithm fbp --output outputs/industrial_ct_fbp
```

`--params` 接受 JSON 对象，外层迭代使用 num_iterations 或 max_iter。安装包包含四份 C++/CUDA 源文件与 BSD 许可证，示例数据保留在仓库 examples/ct_data。

来源卡片推荐预算保存在 configs/ct/industrial_baselines.yaml，由 toy_experiment 使用。SIRT 为 200 次、非负约束，Landweber 为 400 次、非负约束，CG 为 20 次、lam_rel=0.001、tolerance=1e-6。直接公共任务继续沿用 configs/algorithms 的既有默认值。CG 的停止判据沿用本项目正规残差规约，来源相对初始残差的 tol 不能用于声称逐次停止一致。


## 有效几何与步长诊断

三维平行束 FBP 使用最终 ASTRA 体积几何计算各轴体素尺寸，即 WindowMin/Max 的跨度除以对应 GridCount。归一化使用实际等方体素尺寸的平方，window 覆盖后修改名义 voxel_size 不改变相同物理几何的重建结果。平行束 FBP 和 FDK 均要求最终体素各向同性。

公共三维数组的 Y 轴按世界坐标从大到小排列。原生 ASTRA 三维数据在解析重建边界进行 Y 轴转换，匹配迭代投影沿用公共轴约定。工业 FDK 复用 ASTRA 自有缓冲区执行。扇束 FBP 将偏移探测器采样插值到中心探测器，并在旋转轴居中的各向同性网格上执行解析重建，再采样到请求的 window。该解析路径适用于声明的圆轨道平板探测器，其他矢量轨道返回明确错误。插值会带来离散近似误差。

无约束 Landweber 的 convergence.metadata 使用 step_basis 表达实际步长来源。power_iteration_estimate 表示无权重、无正则自动步长，step_norm_squared 为实际平方范数估计，step_estimator_iterations 为 12，非零估计时 step_safety_factor 为 0.9。upper_bound 表示加权或增广目标的 Schur 上界步长，step_norm_squared 与 augmented_squared_norm_bound 相同。explicit 表示显式步长，step_norm_squared 为空，仍保留用于校验的上界。平方范数为零时自动步长回退到 1，实际 step_size 始终报告。

NaSn Siddon 内核以非负射线段长累加，二值探测器掩膜保持其非负性。TomoPyCUDARadon2D 声明 nonnegative_matrix=True，继承的 Schur 上界通过 A1 与 A*1 计算，适用于超过 4096 像素的图像。

旧规则 cone_3d 的 ASTRAFDKOperator3D.fdk 同样在返回边界将 ASTRA 升序 Y 转换为公共降序世界 Y。共享底层 run_fdk_projection 返回原生 ASTRA 轴序，由两个 FDK 入口各自转换一次。对应独立非对称物理检查位于 tests/test_industrial_physics.py。

## 深度算法接入

solvers/deep 与 CT 公共任务的模型资源、公开参数、随机流和尺度说明见 [CT 深度先验](deep_priors.md)。工业投影离散化约定及 NaSn CUDA 编译依赖按本页说明。
