# CT 算法与参数

算法 ID、默认值、约束、几何、设备、权重与运行条件由 `inv_framework.ct.api.list_capabilities(context=...)` 的实际注册表返回。基础方法默认配置位于 configs/algorithms，来源 toy 推荐预算位于 configs/ct/industrial_baselines.yaml。

- [方法索引](../algorithms/README.md)按数值家族组织完整参数与来源卡片。
- [工业 CT](industrial.md)描述平行束、扇束、三维平行束、锥束与截断。ASTRA 工业几何使用 CUDA，解析与匹配离散迭代路径分别说明。
- [深度先验](deep_priors.md)包含 DIP、INR、像素扩散、潜空间扩散及 pMF/LoRA。公共路径为 float32 二维 CT，发布先验须匹配网格、通道、预测类型与尺度。
- [NaSn](nasn.md)说明预处理、训练、EMA 导出及独立重建。
- [Harness](harness.md)说明能力发现、资源暂存、外层预算及部署条件。

DIP/INR 使用内置网络即可拟合。其他深度方法需要本地匹配权重。Budgeted DPS 的子集任务条件和代码接口交付范围见深度先验页。
