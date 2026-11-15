# CT 深度先验与扩散重建

公开算法为 dip、inr、dps、mcg、pgdm、ddim_dds、dmplug、reddiff、hybridreg、psld、resample、diffstategrad、pmf_dds、pmf_lora_dds、budgeted_dps。它们使用 CT API、run_experiment 和 inv-run，算法库依赖保持框架无关。默认参数位于 configs/algorithms，算法说明位于 docs/algorithms。

## 独立接口

```python
from inv_framework.solvers.deep import DIPSolver, INRSolver
from inv_framework.solvers.deep.diffusion import (
    DPSSolver, MCGSolver, PGDMSolver, DDIMDDSSolver, DMPlugSolver,
    REDDiffSolver, HybridRegSolver, PSLDSolver, ReSampleSolver,
    DiffStateGradSolver, DDSSolver, LoRAAdaptiveDDSSolver,
)
from inv_framework.solvers.deep.cg import cg_data_consistency_grad
```

直接构造器保留来源语义。DIP 支持 model_factory、callback、可选 EMA，默认 2000 次、ema_decay=None，卡片推荐为 3000 次和 0.99。INR 为每个 batch 样本独立初始化并拟合网络。DPS 接受 generator、自定义 ConditioningMethod，并传递 step_index、num_steps 和 timestep。DDS 的 MeanFlowSchedule 使用 velocity 预测，DDIM-DDS 使用 epsilon 预测。LoRA 注入、可训练参数组、优化器工厂、内层更新、scale 切换及 save_lora_state/load_lora_state 保留。

CG 数据一致性求解 (A*WA+λI)x=A*Wy+λx_prior。cg_data_consistency_grad 保留梯度图，LoRA 内层使用该路径。各 DDS 变体的方程、batch 规约及停止基准维持来源定义。传入非伴随 back_op 时，标准 Hermitian CG 的保证取决于实际系统。

## 物理域与模型域

公共深度 CT 路径接收 float32、二维图像 (B,C,H,W) 和观测 (B,C,角度,探测器列)。Torch 平行束、ASTRA 扇束以及 NaSn 平行束使用既有投影器。所有模型须匹配任务网格与通道，潜空间模型须匹配 codec。公共结果恢复物理域。

线性尺度声明采用 x_phys=image_scale×x_model+image_offset，输入观测转换为 (y_phys−A(image_offset×1))/image_scale。image_scale 必须为正，默认 1，image_offset 默认 0。独立求解器输入维持调用方声明的模型域。DIP 和 INR 通常直接拟合物理域。

来源 toy 包声明了 image_scale/image_offset 时，examples.run_deep_ct 自动读取，显式参数优先。pMF toy 的典型设置为 0.5 和 0.5，真实扫描使用对应标定文件。来源投影坐标约定及当前工业离散化差别见 industrial.md。来源 pMF toy 是先验自身生成的样本，其质量记录适用于该来源条件。

## 模型资源

安装扩散可选依赖可使用 pip install -e ".[diffusion]"。求解器通过本地路径加载资源。显式下载工具为 tools/download_deep_weights.py，可选择 lodoind-pixel、lodoind-latent 或 pmf。

| 资源 | 固定版本 | 本地布局与要求 |
| --- | --- | --- |
| jiayangshi/lodoind_pixel_diffusion | 0f4a0dcf5768abec78fdbc01b95a83194175c0f1 | weights 目录含 unet/、scheduler/，DPS、MCG 使用 DDPM，其余像素方法使用 DDIM |
| jiayangshi/lodoind_latent_diffusion | 6f35a9b0198a3a7e6eec64b69112745b3a24a295 | weights 目录含 unet/、scheduler/、vqvae/，用于 PSLD、ReSample、DiffStateGrad |
| NaSn pMF EMA | model-v0.1.0，epoch449 | weights 为 safetensors 文件，同名 JSON 位于相邻位置，发布模型为 2048²、单通道、mean-flow velocity |

pMF 推理加载位于 solvers/deep/models/weights.py，模型与发布清单位于相同模型层。完整 NaSn 应用的训练、预处理和导出由 NaSn 独立应用负责。模型、来源代码和数据许可分别见 THIRD_PARTY_NOTICES.md、LICENSES 和 examples/ct_data/README.md。LoDoInd 数据许可为 CC BY-NC 4.0。

## 预算、随机性与结果

外层预算分别为 DIP/INR 的 num_iterations、DMPlug 的 epochs、其他采样器的 num_inference_steps。DMPlug 的采样轨迹长度、CG 内迭代、SIRT 伪逆次数、LoRA adapt_K 和潜域优化预算独立保留。固定预算完成返回 max_iterations 与 fixed_budget_completed，converged 为 false。当前来源循环缺少逐步公共轨迹，诊断明确返回 trajectory_available=false，公共阈值策略无法建立数值收敛结论。

DiffStateGrad 的 pixel_maxiters 和 latent_maxiters 分别控制每次像素域、潜域一致性优化的最大更新次数，默认 2000 和 500，与来源行为一致。两项参数可通过独立构造器、solve、公共 CT 请求及文件任务传入。外层 max_iterations 映射到 num_inference_steps，内层预算保持调用方提供的值。各内层优化保留来源的误差阈值及提前停止条件。

公开参数约束采用显式类型、可空性与上下界。DIP 的 ema_decay 支持 None 或 [0,1)，REDDiff/HybridReg 的 sigma 支持非负值，配置推荐值为 1e-4，原生构造器默认值为 0。pmf_dds 和 pmf_lora_dds 的公开 num_inference_steps 为 1–20。DPS、MCG 和 PSLD 的来源采样步数约束为 1–1000，实际执行还须匹配本地 scheduler 的训练步数。

诊断保留实际外层迭代、网络调用、采样步数和 LoRA 更新次数，INR 另记录独立拟合样本数。公共调用局部设置并恢复全局随机状态。像素算法使用设备 generator，潜域方法和 pMF 的初始逐样本噪声使用 CPU generator。来源部分算法使用全局 randn_like，独立调用复现时也需重置全局随机种子。

sampling_steps 记录基础反向调度步数，DMPlug 按 epochs 累计每次完整采样。network_evaluations 另计内层优化或重采样引起的额外网络调用。

## Budgeted DPS

代码迁入包含 BudgetedDPSSolver、BudgetedPosteriorSampling、BudgetSegmentAction、子集调度、成本函数及 reformulated_gradient，后者维持独立辅助函数的调用地位。原生接口位于 core/budget.py，支持 source_axis、subset 和 subset_measurement。

公共 schedule_actions 为非空的非负整数数组列表，每项对应一个时间段的 source 索引。key_indices 为可空的非负整数数组。未给出 key_indices 时保留来源等宽段划分。CT 映射仅支持现有 Torch 二维平行束角度子集，且要求无探测器掩膜。其他物理任务可独立提供 SubsettableForwardOperator。能力发现返回不能表达子集语义的原因。

本次 Budgeted DPS 交付覆盖代码、接口映射、参数、文档和来源资源，其迁入状态按该范围记录。

## 执行示例

```powershell
python -m examples.run_deep_ct --input examples/ct_data/toy_parallel_ct_128.npz --algorithm dip --output outputs/deep_dip
python -m examples.run_deep_ct --input examples/ct_data/toy_lodoind_512.npz --algorithm ddim_dds --weights weights/lodoind_pixel_diffusion --output outputs/deep_dds --device cuda
python -m inv_framework.runner --config docs/ct/deep_request_template.yaml --output outputs/deep_request
```

模板中的资源与观测路径需要替换为本地文件。Harness 使用公开 CT 能力记录的 parameter_specs、weights、controls.native 和 run_experiment，桥接与运行条件见 [Harness](harness.md)。短预算执行仅说明流程可用，不能代替推荐预算下的重建效果评价。

## NaSn 应用参数

pMF 公共任务支持可空 model_config 显式本地配置路径，runner 按任务文件位置解析。pmf_dds 增加 warm_start=false、t_start=1.0，热启动使用底层 Gridrec 且要求0<t_start<1。完整数据、训练、EMA导出和切片调用见 [NaSn 应用](nasn.md)。
