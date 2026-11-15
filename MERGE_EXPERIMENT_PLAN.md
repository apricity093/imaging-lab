# 成像算法合并与实验方案

本项目以 imaging-lab 的架构、CT／MRI／反卷积领域接口和统一运行入口为基础，整合 inv_framework 的工业 CT、深度先验、扩散重建与 NaSn 应用。新增算法通过公开能力查询和任务接口供 inverse-agent-harness 调用，同时保留独立 Python 调用、命令行、算法卡片、默认参数、示例数据和模型使用说明。

## 当前范围与完成状态（2026-11-15）

本次目标是将固定来源分支的实现接入 imaging-lab，整理模块职责，并提供便于上层 Agent 调用的公共接口。来源已有的数据可用性、数值行为、许可证和验证限制随对应实现保留；不追加真实数据获取、完整效果复现、独立授权核实或上层系统部署完善任务。

实验用于确认流程可执行，统一采用 seed=0 和小预算。服务器从当前源码重新执行，不复用本地实验结果。无需多种子、质量排名、相同时间预算比较、完整推荐预算或长周期训练，也不制作压缩包、wheel、sdist 或独立报告作为交付物。算法库保留源码、必要配置、文档和正式小型示例数据，实验权重、缓存和临时产物不纳入仓库。

| 步骤 | 当前状态 |
| --- | --- |
| 1. 固定版本与映射 | 已完成，保留来源提交与接口、参数、资源对应关系 |
| 2. 经典算法与工业 CT | 已完成并复审，涵盖新增几何、截断、非立方体体积、物理尺度和伴随验证 |
| 3. 深度方法迁入 | 已完成并复审；Budgeted DPS 仅按代码与接口接入范围交付 |
| 4. NaSn 应用迁入 | 已完成并复审；保留来源的数据处理、训练恢复、EMA 导出及重建入口与限制 |
| 5. Agent 调用适配 | 库侧公共契约与独立 harness 检出的配套联调已完成并复审；上层仓库发布和版本切换不作为本库完成条件 |
| 6. 文档、配置与资源 | 已完成并复审；采用源码仓库交付，不再生成分发包 |
| 7. 流程验证 | 已完成约定的服务器单种子、小预算验证，并复审本轮修复；不等同于全量测试、效果复现或所有部署方式验收 |

服务器工作目录为 `/home/gaodongxu/autodl-tmp/imaging-lab`，使用既有 `imaging-lab` 环境的 Python，额外依赖与缓存均位于工作目录内。新执行的 18 条公开 CT 流程和 6 条 harness 源码直接调用链通过；发布模型保持原 512²／2048² 网格，NaSn CUDA、Gridrec、合成标定及小型训练恢复／EMA 流程已运行。实际输出保留在该目录的 `runs/public`、`runs/harness`、`runs/native.json` 和 `runs/application`。

真实 NaSn 数据未发布、来源合成跨运行差异，以及上层隔离 FDK／VarNet 部署限制继续说明，不扩展为本次新增开发任务。源码改动尚未推送；任何远程推送均等待用户明确授权。

## 合并范围

| 功能 | 整合要求 | 主要位置 |
| --- | --- | --- |
| 架构与公共任务 | 保留三领域结构、统一 runner、结果契约及独立使用能力 | inv_framework/core、api、runner.py、各领域 api.py |
| FBP、SIRT、Landweber、CG-Tikhonov | 比较重复实现并整合有效差异，保留现有扩展目标与诊断能力 | solvers、core、ct |
| 其他经典、统计、变分与组合方法 | 保留现有实现、配置和公开入口 | 现有求解器及三领域模块 |
| MRI、反卷积 | 保留现有实现，吸收适用的扩展说明和教程内容 | mri、deconvolution、对应文档 |
| DIP、INR、DPS | 以来源分支实现为基准，迁入现有深度模块 | solvers/deep |
| 扩散重建 | 接入 MCG、PGDM、DDIM-DDS、DMPlug、RED-Diff、HybridReg、PSLD、ReSample、DiffStateGrad | solvers/deep/diffusion |
| Mean-flow 与适配 | 接入 pMF DDS、LoRA-DDS、模型、调度器及可微数据一致性校正 | solvers/deep/diffusion、solvers/deep/models |
| Budgeted DPS | 按来源分支迁入原语和求解器，仅进行必要的导入、数据结构和公开接口映射 | solvers/deep/diffusion、必要的共享算子接口 |
| 工业 CT | 扩展截断探测器、扇束、三维平行束、锥束和 NaSn 投影器的完整执行链路 | ct、api、配置与数据 |
| NaSn | 保留数据处理、训练、权重导出、重建和基准为独立应用 | applications/nasn_ct |
| Harness 适配 | 完善库侧发现、参数、输入、权重、预算与结果契约，通过独立上层检出验证调用 | 本项目领域 API；独立 harness 联调代码 |
| 文档与数据 | 随算法完整迁移，更新同名算法文档和领域手册 | docs、configs、examples、tools |

USCT、FTTT 的场景注册、物理求解器、专属数据、策略模型及 Diffusion_Policy 外部任务启动功能不纳入本项目。Budgeted DPS 的迁入不引入这些应用依赖。

## 模块组织

沿用现有模块职责。共享算子、保真项、正则和数值基础位于 core；CT 几何、观测、案例和投影后端位于 ct；经典与变分求解器位于 solvers；深度算法与模型位于 solvers/deep；NaSn 编排位于 applications/nasn_ct。MRI 与反卷积继续由各自领域模块管理。

新增内容采用以下归属：

- CT 投影器及 CUDA 源文件归入 inv_framework/ct/backends，截断和采样操作归入 CT 算子／采样模块。
- 扩散算法、调度器、conditioning、LoRA 与相应模型归入 inv_framework/solvers/deep。
- Budgeted DPS 所需的 source-subset 算子接口保持共享职责，连接已有 CT 角度子集能力时只做必要映射。
- NaSn 数据预处理、标定、训练、模型导出和批量应用流程归入 inv_framework/applications/nasn_ct。
- 默认参数沿用 configs/algorithms、configs/mri/algorithms、configs/deconvolution/algorithms 的现有布局。
- 工业 CT 示例、配置及小型数据按用途组织，训练权重和完整数据集通过明确的本地路径或独立下载工具提供。

迁移不维护两份同名数值实现，不让算法库依赖 harness SDK。已有模块导入和独立示例需要的入口在原模块位置继续提供；仅对实际存在的调用差异做参数映射。

## 经典与变分算法整合

### FBP 与 FDK

二维平行束 FBP 保留一份 Ram-Lak 滤波、补零、反投影和角度缩放实现。来源分支提供的扇束 FBP、三维平行束 FBP 与锥束 FDK 按各自几何接入。

公共算法身份继续区分 fbp 与 fdk。来源卡片中 fbp 按几何调用 FDK 的运行方式映射到明确的 fdk 任务，卡片保留原始来源及更新后的命令。过滤器、几何尺度、短扫描等选项仅在相应后端实际支持时公开。

### SIRT 与 Landweber

SIRT 保留行列归一化、初始化和盒约束行为，整合来源低层实现对观测 dtype 的保持，继续提供本项目已有轨迹、停止和资源诊断。

Landweber 保留本项目加权二次目标、增广正则和有界路径，接入来源工业 CT 配置所需的自动步长。来源的幂迭代范数估计属于步长估计，不作为严格上界声明。旧实验使用明确参数固定原行为；新推荐配置写明自动步长规则。参数选择及详细结果应与低层调用一致。

### CG 与数据一致性

区分零中心 Tikhonov 与带先验中心的数据一致性：

`(A* A + λI)x = A* y`

`(A* W A + λI)x = A* W y + λ x_prior`

来源 cg_tikhonov 的 lam_rel 按其定义换算为 λ = lam_rel × ||A||²，并同时记录输入相对权重与实际权重。不得将 lam_rel 直接当作本项目已有绝对正则参数。

LoRA-DDS 保留可微 CG 校正。现有共享 CG 的 no_grad 路径继续用于其原有任务，不替代需要梯度传播的校正。DDS 参考实现中不同 CG 方程、停止基准和 batch 规约按算法分别保留；非伴随反投影选项不赋予标准 Hermitian CG 的数学保证。

ADMM、PDHG、FISTA、SART、OS-SART、MLEM、OSEM、约束 TV、CGLS、LSQR 和三领域组合求解能力保持现有契约，修改仅限整合所需的共享接口。

## MRI 与反卷积

保留 MRI 的复数表示、Hermitian 伴随、Cartesian／非 Cartesian 编码、多线圈校准及重建能力，保留反卷积的边界、PSF 原点、噪声模型、已知核及盲估计能力。

来源的 Fourier、卷积和非线性扩展示例用于完善接口教学，示例调用采用本项目现有算子。MRI 教学示例的 Fourier 伴随缩放按空间维度定义，不使用包含 batch 的元素总数。教学骨架不替换正式领域实现。

本次新迁入的 CT 先验不自动注册为 MRI 或反卷积算法。跨领域共享方法文档更新时，保留各领域既有的目标、参数、复杂数及数据范围说明。MRI／反卷积不增加与本次合并无关的扩散效果实验。

## 深度求解器与 Budgeted DPS

DIP、INR、DPS 及其配套网络以来源分支为实现基准。DIP 保留 EMA、model_factory 和 callback；构造默认值与卡片推荐值分别表达。INR 保留每个观测样本独立拟合网络的行为。DPS 保留随机生成器、观测 dtype、采样步骤上下文及自定义 conditioning。

像素域 epsilon 先验、潜空间先验和 mean-flow velocity 先验分别匹配其模型、codec、调度器与数据尺度。DDIM-DDS、pMF-DDS 和 pMF-LoRA-DDS 使用明确的算法 ID 和配置。LoRA 保留原注入位置、可训练参数、优化器参数、内层更新及保存加载能力。

Budgeted DPS 按来源分支接入 BudgetedDPSSolver、BudgetedPosteriorSampling、子集调度、成本辅助函数、reformulated_gradient 及必要的 source-subset 算子接口。保持原语之间现有调用关系；独立辅助函数不会因接入而自动成为默认引导步骤。

Budgeted DPS 的范围限定为代码迁入、必要接口映射、参数描述、算法说明和来源资源保留。面向 harness 时，显式的调度参数以可序列化形式映射到原生结构；具体任务只连接已有且能表达所需子集语义的算子，不引入 USCT／FTTT 外部进程。不能满足子集接口的任务在能力查询中明确说明原因。

不为 Budgeted DPS 新增专项测试、对照实验、性能实验、质量排名、角度策略、策略学习或归一化梯度变体研究。本方案其他实验组不扩展到 Budgeted DPS；迁入状态不表述为新增数值或性能验证结论。

## 工业 CT 与 NaSn

工业 CT 接入覆盖请求解析、能力查询、几何构建、观测加载、投影后端、求解器和评价。任务区分二维平行束、二维扇束、三维平行束和三维锥束，探测器截断作为采集条件显式表达。

几何字段包含图像／体积形状、观测形状、角度、体素及探测器尺寸、旋转中心、距离、偏移和所需矢量几何。支持来源示例的非立方体体积，不通过修改数据分辨率绕过既有立方体限制。图像宽度与探测器宽度分别声明。

NaSn 应用保留数据获取、目标扫描和 FullCT 预处理、尺度标定、训练集构建、训练及恢复、EMA 导出、重建和结果保存。单次重建接入公开 CT 任务；预处理与训练保留专用 CLI，不强行映射为普通重建任务。

来源分支的 NaSn 数据清单为 `published=false`，真实重建需要调用方已有的本地扫描、标定文件及匹配权重。本库沿用该条件，不负责发布或补齐原始扫描。来源已记录真实切片迁移对照通过、合成跨运行严格对照未通过，以及完整训练和完整真实 LoRA 实验未重跑；这些记录保持来源归属，不作为本库的新实验结论，也不要求本次补做。

CUDA 扩展、Gridrec、ASTRA 和模型依赖按实际使用惰性加载，保持基础 CPU 入口可用。源码仓库包含所需 CUDA 源码、配置、清单和来源许可证，供获取源码后直接调用。

## Harness 接入契约

### 调用链与职责

本项目对上层提供以下公共接口：

- inv_framework.ct.api、inv_framework.mri.api、inv_framework.deconvolution.api 的能力查询和领域任务入口。
- inv_framework.capabilities 的聚合能力查询。
- inv_framework.runner.run_experiment 及 inv-run 的统一文件任务与结果输出。

CT 的 api.py 继续导出 inv_framework/api 中的请求、校验和执行能力。inv_framework/ct/backends 管理 Torch、ASTRA、LEAP、NaSn 等数值投影后端，不承担 harness 请求编排。

上层 harness 的适配位置是 src/inverse_research/backends/imaging.py、adapters.py、parameters.py、staging.py 及相应实验执行模块。三领域共用 imaging-lab 的绑定，保留其 provider_id=ct 约定。这些属于上层仓库，算法库不导入或依赖其 SDK。

调用顺序为：

`harness 能力查询 → 参数及任务准备 → 数据与权重暂存 → 领域校验 → run_experiment → result.json 与数组 → harness 评价及诊断`

本库交付以公共接口可被上层正确调用为准，不能仅以投影后端可运行代替调用链验证。已完成的 harness 改动位于独立本地检出 `outputs/inverse-agent-harness` 及服务器工作目录的 `harness` 中，用于配套联调；它们没有并入算法库源码，也没有推送上游。后续上层可从 GitHub 获取算法库源码；上层仓库的正式发布、固定提交切换和部署管理不列为本次算法库接入的未完成项。

### 能力发现与参数

新增算法进入真实注册表、默认配置和领域能力查询。元信息至少覆盖算法身份、实现状态、任务类型、输入输出、几何、维度、设备、依赖、权重、参数及合法范围、迭代控制和运行条件。

更新上下文查询，使权重、实际设备、几何、截断条件、模型分辨率和可用输入参与适用性判断。查询不加载大模型或启动重建；缺少运行资源时仍能描述算法，并返回缺失条件。

参数类型、可空值、枚举、上下界和原生默认值传递给 harness。嵌套调度、数组参数和模型配置的必要结构由两侧共同表达，不能只声明 object 或 array 而丢失实际约束。配置文件、直接 Python 调用和 harness 请求的参数解析保持一致。

联调时使上层目录发现、绑定信息及必要展示能够看到新算法和几何。元信息缓存随实际源码版本刷新，不通过额外硬编码一份默认参数来实现接入。

### 预算与随机性

通过 controls.native 声明实际外层预算字段。DIP／INR 的训练迭代、DPS／DDS 的采样步数、DMPlug 的外层优化次数分别映射；CG 内迭代、LoRA 内层步数和潜空间优化预算作为独立参数保留。harness 的 max_iterations 不同时覆盖多个内外层循环。

保留固定步采样与阈值收敛的区别。记录实际采样步、内层更新、网络调用、残差、耗时与停止原因。种子按来源算法和设备要求传递，CPU 与设备随机生成器的约定在文档中明确。

### 后端与几何传递

调整 harness 中 CT 请求构建和执行桥接的固定 Torch 选择，使用户选择的数值后端贯穿准备、校验和运行。扩展 CT staging 的几何字段，保留扇束、三维平行束、锥束及截断所需的数据。

保持目录案例入口，并补齐新增外部观测或 toy 数据到公共任务的确定映射。两侧不能分别推断不同的几何、尺度或后端。harness 的部署可用性与算法库原生能力分别记录。源码直接调用 FDK 已通过；上层 Linux 隔离执行的 FDK 门禁保留，不将解除该门禁扩展为本次算法库任务。

### 权重与模型资源

为 UNet、scheduler、VQ codec、pMF 权重及相邻配置建立明确的资源声明和路径传递。源算法使用目录权重时保留其目录结构。模型网格、预测类型、通道和物理尺度需与实际任务匹配。

harness 在运行前准备所需资源，并将运行环境可访问的本地路径传入算法库；求解过程中不隐式下载。正确传递配置相对路径和运行环境，保留 CUDA 扩展构建所需源码。已有 VarNet 及本次新增深度算法的资源准备分别说明，不因其他算法接入而默认解除部署限制，也不追加 VarNet 部署改造。

### 结果与评价

保持 ExperimentResult、result.json、reconstruction.npy、output_artifacts 和既有错误语义。附加算法诊断通过现有 diagnostics 表达，保存最终有效参数、实际设备和计算精度。

harness 暂存的零真值占位数据仅用于原有目录格式，不能用于图像质量证据。参考图像仍只用于重建后的独立评价。原生执行成功、预算结束、数值收敛和部署阻塞分别表达；能力发现成功不等于部署可执行。

## 数据与数值语义

| 项目 | 执行约定 |
| --- | --- |
| 观测域 | 线积分、计数及其他支持域显式声明；来源 toy sinogram 不再次取负对数 |
| 缺测 | 显式传递探测器掩膜及其算子作用；不将所有数值零统一解释为缺测 |
| 模型尺度 | 保存 image_scale、image_offset，明确模型域与物理域转换 |
| 二维与三维布局 | 分别声明 batch、通道、角度、探测器行列和体积轴 |
| 伴随 | 分别建立离散伴随与 autograd 的一致性，不以自定义 backward 的互相调用替代数学依据 |
| 数据一致性 | 区分 Landweber 数据步、SIRT 投影、解析初始化和各类 CG 校正 |
| dtype | 保留原有复数和精度契约；新增后端声明实际精度，不静默扩大支持范围 |
| 评价 | 公共物理域指标与原卡片指标分列，写明 ROI、动态范围、SSIM 定义和裁剪规则 |

线性观测的尺度转换采用：

`x_phys = s × x_model + b`

`y_model = (y_phys − A(b × 1)) / s`

偏置项必须随观测转换。该公式不直接用于非线性模型。原有独立模块使用自己的已声明物理约定，不因深度模型输入范围改变而被统一归一化。

## 算法文档与领域手册

### 目录职责

| 路径 | 职责 | 本次更新 |
| --- | --- | --- |
| docs/algorithms/ | 按数值方法／家族组织的算法说明 | 新增深度方法页面，更新同名方法及索引 |
| docs/ct/ | CT 任务、算法适用性、数据格式和配置模板 | 补齐工业几何、深度重建、模型资源和调用 |
| docs/mri/ | MRI 任务、复数与采样模型、领域算法及模板 | 同步受共享接口影响的说明，保留领域契约 |
| docs/deconvolution/ | PSF、边界、噪声、盲估计及配置模板 | 同步受共享接口影响的说明，保留领域契约 |
| docs/agent_algorithm_api.md | 框架无关的任务、能力、控制和结果契约 | 补充新算法及 harness 接入字段 |
| docs/architecture.md | 模块职责与调用关系 | 区分数值后端、领域接口与 harness 适配 |
| docs/README.md | 文档导航 | 更新算法、领域手册和应用入口 |
| tools/render_algorithm_docs.py | 按注册信息生成方法说明 | 扩展方法元信息和输出，保留完整卡片内容 |

docs/ct、docs/mri、docs/deconvolution 与 docs/algorithms 同级。新 CT 算法同时提供方法页和 CT 领域接入说明，不创建空的 MRI／反卷积算法条目。

### 算法页面

新增 DIP、INR、DPS、MCG、PGDM、DDIM-DDS、DMPlug、RED-Diff、HybridReg、PSLD、ReSample、DiffStateGrad、pMF-DDS、pMF-LoRA-DDS、Budgeted DPS 的对应说明。CG-Tikhonov 在 CG 家族页面保留完整参数及来源卡片映射。

更新 FBP、FDK、SIRT、Landweber、CG 等同名页面，保留同一页面中的 MRI、反卷积领域内容。每个纳入算法说明包含：

- 身份、数学定义、适用条件、支持几何与实现位置。
- 输入输出、单位、形状、尺度、设备与模型要求。
- 原生参数、推荐预设、参数来源及覆盖优先级。
- 独立 Python、文件任务、CLI 和 harness 调用方式。
- 配套数据、权重、来源、许可、评价口径和适用限制。

来源已有卡片的有效内容完整保留，运行路径更新为整合后的入口。算法文档生成器同时承载参数元信息和卡片补充正文，避免重新生成时丢失来源、模型或数据说明。算法数量和索引由实际注册内容更新，不沿用固定旧计数。

### 领域文件

docs/ct/algorithms.md 增加新算法、几何、设备和权重适用条件；tasks.md 增加工业及深度重建任务；data_format.md 增加扇束、三维平行束、截断、NaSn 和模型尺度约定。

保留现有 request_template.yaml，补充以用途命名的工业 CT、扩散和 NaSn 请求模板及可运行配置。模板明确区分模型文件、观测输入、算法参数和评价参考。Budgeted DPS 文档描述原生子集接口和显式调度，不附加新实验或质量结论。

docs/mri 和 docs/deconvolution 中 algorithms.md、tasks.md、data_format.md、request_template.yaml 仅随实际共享接口变化更新；领域专属数据规范不因 CT 扩展而改写。

### 配套资源

保留来源的八类 toy 数据及其说明：二维平行束、截断平行束、三维平行束、三维锥束、LoDoInd 平行束／扇束、pMF 截断／扇束。保留生成脚本、预处理、参数、权重获取说明、已知验证记录和许可。

实验所需权重、缓存、临时文件和运行输出存放在服务器指定工作目录内，独立于算法库源码。保留可解释的实际运行结果即可，不新增独立报告或分发包。本地实验下载的权重及重复缓存、构建临时产物已清理，相关目录不纳入 Git。模型生成的 pMF toy 数据用于匹配先验流程验证，不作为真实工业泛化能力的主要证据。

## 实验与验证

Budgeted DPS 不纳入本节的新增测试、实验或性能评价。其他算法保留已有迁移一致性和数学契约验证，新增服务器实验用于确认单种子、小预算下的流程可执行，避免算法与几何的无意义全组合。

| 实验组 | 输入与对照 | 观察内容 | 验收条件 |
| --- | --- | --- | --- |
| 原功能回归 | 现有三领域及组合示例、相关项目检查 | 参数、输出、复数、停止语义 | 原有独立功能和契约保持可用 |
| 经典实现整合 | 同一算子、输入、初始化及显式参数下的双方实现 | 更新、残差、尺度与重建 | 重复实现差异有明确语义，整合结果符合所选定义 |
| 工业几何 | 来源各新增几何及截断数据 | 几何方向、伴随、单位、布局、非立方体体积 | 公共任务完整执行，数学和数据契约成立 |
| 深度求解迁移 | 相同配置的 DIP／INR／DPS | EMA、逐样本拟合、随机流、引导 | 保留来源行为和公共输出语义 |
| 扩散重建 | 已有小模型来源对照，以及发布权重短预算运行 | 调度器、重建、残差、调用数 | 来源行为对照有依据，匹配权重流程可执行 |
| Mean-flow 与 LoRA | 小模型及代表性 pMF 任务 | 校正梯度、适配参数、保存加载、显存 | 可微路径与实际推理成立 |
| NaSn 应用 | 来源支持的小型合成数据流程、短训练及匹配先验调用 | 原生投影、标定、恢复、导出、结果 | 可用资源下流程可执行，真实数据条件按来源说明 |
| Harness 联调 | 通过实际发现、暂存、校验、执行和结果消费链路提交任务 | 后端、几何、权重、预算和诊断 | 支持的算法可被上层正确调用，部署条件如实表达 |
| 源码入口与文档 | 源码公共接口、CLI 及卡片命令 | 资源、入口、依赖、链接、配置 | 获取源码后可调用，文档与实际接口一致，无需分发包 |

### 对照与计算预算

经典方法先固定显式参数比较更新，再比较推荐配置；Landweber 分开比较固定步长和自动步长。新几何先用 FBP／FDK 与一个迭代基线确认尺度和布局，再运行匹配深度方法。

扩散迁移对照固定数据、权重、预处理、模型预测类型、调度器、随机设置和计算预算。来源已有 CPU 参考测试适用时沿用 rtol=1e-5、atol=1e-6；实际 CUDA 后端和大模型使用各自数值协议，不将 CPU 门槛推广到全部路径。

每个纳入实验范围的公开算法至少有一个匹配数据、模型和维度的端到端案例；每个新增几何覆盖其对应后端。INR 包含不同观测样本的 batch 案例。二维先验不强制用于三维体重建。

服务器流程验证固定 seed=0，采用足以触发实际执行路径的短预算，内外层参数分别保留。记录实际采样步、内部更新、网络调用、耗时、设备和停止状态，按方法保留适用残差诊断。无需多种子波动、推荐完整预算、相同时间预算比较、质量排名或大规模调参；已有来源效果数据继续注明其原始条件。

### Harness 验收

以本项目真实 API 结果为能力来源。至少覆盖 CPU 基础任务、一个新增 CUDA 几何、像素域权重、潜空间权重及 pMF 资源路径，并分别检查外层预算和内层参数。

验证配置相对路径在暂存后的含义、模型目录结构、几何字段保留、数值后端选择、结果数组和诊断消费。真实参考数据由 harness 的评价流程使用，不进入重建参数或初始化。

本项目原生可运行但某种上层部署未完成的任务保留具体说明。直接调用、隔离调用分别记录，不互相代替。当前完成的是源码直接调用链验证，不要求补齐 Linux 隔离门禁或其他上层部署方式。

### NaSn 验收

短训练验证数据流、参数更新、保存恢复和 EMA 导出；推理验证匹配权重、原生算子及代表性可用数据的调用。真实扫描的获取条件、真实标定和无配对真值等限制沿用来源说明。完整长周期训练、全部真实切片和来源效果复现不属于本次完成条件。

没有真实参考图像的任务报告观测残差及适用诊断，不生成无依据的 PSNR／SSIM。来源已有验证限制保留在应用文档和来源记录中。

## 实施顺序

1. 固定双方代码版本和相关调用契约，落实功能对应表、参数及资源映射。
2. 整合重复经典方法，扩展工业几何、观测数据及公共 CT 任务。
3. 迁入 DIP、INR、DPS、扩散、mean-flow、LoRA；Budgeted DPS 按限定范围原样接入。
4. 迁入 NaSn 应用，完成模型资源和训练／重建命令。
5. 完善本项目能力查询和调用契约，通过独立 harness 检出的发现、暂存、参数、预算、权重及执行桥接验证。
6. 完成算法文档、领域文件、卡片资源、配置模板及源码调用入口。
7. 执行约定的单种子、小预算流程验证与相关检查，修复合并导致的失败，在会话汇报结果与保留限制。

代码、配置、卡片和领域文档随对应能力一起落地；数值算法迁入后仍需完成公开接口和适用的 harness 适配，才计为对应功能交付。

## 交付与适用条件

本次交付为算法库源码及其必要的配置、文档、原配套小型数据和工具，以及可被上层调用的公共契约。独立 harness 联调改动与实际服务器结果保留在各自位置，不将上层框架纳入算法库，不新增压缩包、wheel、sdist 或独立交付报告。会话汇报实际环境、通过路径与来源限制，不以跳过代替通过。

基础源码入口继续支持现有 CPU 任务。扩散依赖采用来源实现适用的版本组合；NaSn、训练、ASTRA 等保持可选依赖。数据、代码、调度器和模型分别保留来源的许可、署名及第三方说明。来源未提供独立许可证的材料，沿用来源对此的表述，不自行赋予新许可，也不追加独立授权核实任务。

服务器已取得匹配发布模型并完成原生 CUDA 流程验证。本地 Windows 阶段的工具链与权重缺失记录只说明当时环境，不再表述为服务器当前阻塞。真实 NaSn 数据访问限制、来源合成跨运行对照限制及 CUDA 原子累加波动继续按来源记录表达，不通过放宽门槛掩盖差异。Budgeted DPS 交付仅声明代码与接口接入。

当前约定的迁入、适配和流程验证工作已完成；未确认存在需要继续实施的功能缺口。真实数据发布、完整训练和质量复现、上层隔离部署、上层仓库版本发布均不列为待办。后续如需 GitHub 发布，由用户另行明确授权；获得授权前保持未推送状态。

## 参考入口

- [imaging-lab](https://github.com/apricity093/imaging-lab)
- [工业 CT 与算法卡片来源分支](https://github.com/peterw634/inv_framework/tree/yuan_cx/inv_framework_indus_ct_cards)
- [来源算法注册表](https://github.com/peterw634/inv_framework/blob/yuan_cx/inv_framework_indus_ct_cards/inv_framework/registry.py)
- [来源扩散方法](https://github.com/peterw634/inv_framework/blob/yuan_cx/inv_framework_indus_ct_cards/docs/diffusion_methods.md)
- [来源 Budgeted DPS](https://github.com/peterw634/inv_framework/blob/yuan_cx/inv_framework_indus_ct_cards/inv_framework/solvers/diffusion/budget_dps.py)
- [来源示例数据](https://github.com/peterw634/inv_framework/blob/yuan_cx/inv_framework_indus_ct_cards/examples/data/README.md)
- [来源 NaSn 应用](https://github.com/peterw634/inv_framework/blob/yuan_cx/inv_framework_indus_ct_cards/docs/nasn_ct.md)
- [来源第三方说明](https://github.com/peterw634/inv_framework/blob/yuan_cx/inv_framework_indus_ct_cards/THIRD_PARTY_NOTICES.md)
- [Harness 成像适配](https://github.com/lucian-dw/inverse-agent-harness/blob/a0e672e11cf29ca5b79f9e98f7f3a6acf94089df/src/inverse_research/backends/imaging.py)
- [Harness 数据暂存](https://github.com/lucian-dw/inverse-agent-harness/blob/a0e672e11cf29ca5b79f9e98f7f3a6acf94089df/src/inverse_research/backends/staging.py)
- [本项目公共任务契约](docs/agent_algorithm_api.md)
- [本项目算法索引](docs/algorithms/README.md)
- [本项目领域手册](docs/README.md)
