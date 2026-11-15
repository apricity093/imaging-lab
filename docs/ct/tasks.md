# CT tasks

Reconstruct attenuation from explicit parallel-2D, fan-2D, parallel-3D or cone-3D acquisition data.
Use `inv_framework.ct.api` for discovery, validation and one solve. The existing
catalog route remains optional; external arrays/files use `CTTaskRequest.inputs`.
Truth is optional evaluation data. No reconstruction parameter is inferred from
truth. [Public API](../agent_algorithm_api.md) describes statuses and stopping.


工业 CT 几何、来源版本、经典方法参数映射及资源约定见 [工业 CT 整合契约](industrial.md)。

## CT 深度重建

DIP、INR、DPS、扩散重建、mean-flow 与 LoRA 的公共任务、模型资源、尺度和预算见 [CT 深度先验](deep_priors.md)。Budgeted DPS 的显式子集映射与交付范围同时记录。

工业、扩散与 NaSn 的文件任务模板和资源替换说明见 [请求与安装资源](resources.md)。NaSn 预处理与训练使用专用 CLI，单次重建使用公开 CT 请求。
