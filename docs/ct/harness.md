# Harness 接入

上层适配参考 lucian-dw/inverse-agent-harness 的提交 `a0e672e11cf29ca5b79f9e98f7f3a6acf94089df`。三领域共享 imaging-lab 源码快照和 Python 环境，安装身份保持 `provider_id=ct`。

## 安装与发现

只读 Python 环境配合独立依赖目录时，通过启动 Harness 的 PYTHONPATH 提供依赖。数值子进程将所选源码目录置于首位，并继承依赖路径、PYTHONDONTWRITEBYTECODE、CUDA 编译环境及显式缓存目录。安装回执按实际路径优先级记录依赖版本。缓存目录应按部署工作区设置，模型仍通过本地资源参数暂存。

通过 harness 的 attach 流程连接当前 imaging-lab 检出及运行环境。安装时从实际导出的源码查询三领域能力，生成带源码版本与适配器版本的目录快照。适配器版本为 v2，旧快照会重新发现。完整参数、默认值、枚举、边界、可空性及嵌套数组结构来自公共能力查询。

`base` 保留 CPU 基础依赖，`diffusion` 和 `nasn` 分别选用源码声明的可选依赖。ASTRA、CUDA Toolkit/MSVC 及模型发布权重仍需按任务部署。选定源码未声明的安装 profile 会被拒绝。仓库默认远程安装版本仍由 harness 原有 lock 文件固定，本步骤未发布 imaging-lab 提交，使用本次成果须 attach 本地检出。

`inv_framework.capabilities.list_capabilities(modality, context=...)` 与各领域能力查询接收同一上下文。CT 上下文包含 `params`、`backend`、`device`、`model.geometry`、`model.measurement` 和 `available_inputs`。查询只读取声明和本地模型 JSON 配置，模型网格、通道和预测类型参与筛选。潜空间方法使用 UNet sample_size 与 VQ 解码器上采样倍率计算实际解码网格，单层 VQ 的倍率为 1，常规多层倍率为 `2**(len(block_out_channels)-1)`。实际张量兼容性在执行时验证。

## 数据与几何

目录案例继续使用 catalog.json、case.json 和 arrays.h5。harness 案例配置中的 `numerical_backend` 明确指定 torch、astra、leap 或 nasn，缺省沿用目录元数据 `execution.backend`，最终回退 torch。`device` 与 `physics_variant` 显式选择实际设备和几何。

暂存保留体积/探测器形状、角度、距离、voxel_size、det_pixel、det_offset_u、center、vectors、window 和 detector_mask。校验与执行读取同一份暂存元数据。观测保持线积分，显式零值不自动表示缺测。

`Benchlab.import_ct_source(source, destination, case_id, truth_unit=...)` 将公共外部 JSON/YAML 任务或来源 toy NPZ 转为私有目录案例。外部任务采用 `model.geometry`、`model.measurement`、`inputs.observation` 及可选 `inputs.valid_measurement_mask`。文件引用相对任务文件解析。toy 映射复用 imaging-lab 的 `toy_experiment`，保留其后端、中心、截断和尺度。`truth_unit` 默认 normalized_inverse_length，物理单位任务需显式设置。

导入阶段可保存评价参考至私有案例。数值暂存只复制观测和公开物理信息，目录兼容真值为零占位。harness 独立评价器在重建结束后读取原私有参考，原生占位图像指标从科学证据中排除。

来源 toy NPZ 的 image_scale/image_offset 及外部任务 params 中的同名声明保存为 case.json 的 `execution.model_parameters`，公开暂存保留这两个字段。参数准备优先级依次为原生默认值、导入案例声明、显式请求参数；预算和种子继续由运行控制提供。适配器仅向声明支持这些参数的算法传递尺度，解析结果记录 imported_case 或显式请求来源。直接 toy_experiment 调用深度算法时也沿用来源声明，显式 params 优先。

## 模型资源

运行参数 `weights`、`model_config` 采用本地资源。harness 请求中的相对路径相对 `data_root`，案例可用 `model_resource_root` 指定其相对子目录。文件任务直接调用 imaging-lab 时仍相对任务文件解析。

准备阶段解析绝对路径，预检和执行阶段分别复制资源至数值输入目录的 models 下。UNet、scheduler、vqvae 目录完整保留，pMF safetensors 的相邻 JSON 随权重复制，显式 model_config 独立复制。暂存后的路径写入 `numeric/algorithm-staged.yaml`，Linux 数值沙箱通过现有公开输入挂载可访问这些路径。执行过程采用本地加载。

VarNet 的部署限制保留。发布权重缺失、模型不匹配及 CUDA 编译缺失分别返回具体条件。FDK 的 local-trusted 路径使用公开 CT 校验及 ASTRA CUDA 检查，并由实际执行结果决定成功状态。Linux 隔离部署的 FDK 限制保留。

## 预算与结果

`run_controls.max_iterations` 只映射 `controls.native.iteration_parameter`。DIP/INR 使用 num_iterations，扩散与 DDS 使用 num_inference_steps，DMPlug 使用 epochs。CG 内循环、LoRA adapt_K、潜空间优化及 DMPlug 采样次数保留各自参数。种子来自 run.seed，潜空间算法使用 CPU 随机生成器，像素域按实际设备创建生成器。

`result.json`、`reconstruction.npy`、output_artifacts 和 diagnostics 由公共 runner 保存。harness 保留有效参数、实际迭代、网络调用、残差、设备、精度及停止原因。固定预算结束表示执行完成，converged 保持原生数值语义。资源筛选通过的预检记录为 validated_pending_execution。

算法库提供框架无关的公共入口，上层仓库负责调用编排与部署。
