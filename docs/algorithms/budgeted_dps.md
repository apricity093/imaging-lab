# Budgeted DPS

Explicit source-subset DPS conditioning.

## 实现与执行约定

固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78。实现位于 inv_framework/solvers/deep，公开 ID 为 budgeted_dps。

Code and interface migration. Explicit schedules supplied by the caller.



CT 公共任务接收 float32 图像 (B,C,H,W)、观测 (B,C,角度,探测器列)，支持二维平行束与扇束。独立求解器保留原生类型及构造参数。模型域转换、资源与运行条件见 [深度先验](../ct/deep_priors.md)。

## 公共参数

| 参数 | 默认值 | 含义与约束 | 来源与推荐范围 |
| --- | --- | --- | --- |
| num_inference_steps | 200 | Reverse sampling steps {"type": "int", "nullable": false, "minimum": 1, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "positive integer"} | source constructor； |
| scale | 1.0 | Residual gradient scale {"type": "float", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "positive real"} | source constructor； |
| eta | 0.0 | Scheduler stochasticity {"type": "float", "nullable": false, "minimum": 0, "maximum": 1, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": "[0,1]"} | source constructor； |
| schedule_actions | None | Source indices per segment {"type": "subset_schedule", "nullable": true, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "schema": {"type": "array", "minItems": 1, "items": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}}}, "theoretical_constraint": "nonempty array of nonempty nonnegative integer arrays"} | explicit caller schedule； |
| key_indices | None | Optional segment start steps {"type": "int_array", "nullable": true, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "schema": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}}, "theoretical_constraint": "null or nonnegative integer array"} | source constructor； |
| weights | None | Local pretrained model directory or pMF safetensors file with adjacent JSON {"type": "str", "nullable": true, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| seed | 0 | Seed for network initialization and sampling random streams {"type": "int", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| image_scale | 1.0 | Positive physical image scale in x_phys = image_scale*x_model + image_offset {"type": "float", "nullable": false, "minimum": 0, "maximum": null, "inclusive_minimum": false, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |
| image_offset | 0.0 | Physical image offset, included in observation conversion through A(offset*1) {"type": "float", "nullable": false, "minimum": null, "maximum": null, "inclusive_minimum": true, "inclusive_maximum": true, "choices": [], "theoretical_constraint": null} | 公共 CT 参数，默认配置与注册表 |

YAML 默认值和参数含义来自实际注册信息，来源卡片保留推荐值依据。显式参数覆盖任务文件，任务文件覆盖算法配置。直接构造 DIP 默认 2000 次、EMA 关闭，推荐配置为 3000 次和 EMA 0.99。

## 独立调用与文件任务

```python
from inv_framework.ct.toy import toy_experiment
from inv_framework.runner import run_experiment
config = toy_experiment('examples/ct_data/toy_lodoind_512.npz', 'budgeted_dps')
config['params']['schedule_actions'] = [list(range(40))]
# Set config['params']['weights'], image_scale and image_offset for the selected prior.
result = run_experiment(config, output_dir='outputs/budgeted_dps')
```

完整文件请求见 docs/ct/deep_request_template.yaml。使用 python -m inv_framework.runner --config REQUEST.yaml --output OUTPUT 执行，输出 result.json 和物理域 reconstruction.npy。上层调用使用相同 CT API 和声明的外层预算，无需 Agent SDK。

## 验证与资源限制

本方法按代码、接口映射、参数与来源资源记录交付范围。

## 数学与方法差异

DIP/INR 在固定观测下优化网络参数，使 A fθ 与 y 匹配。扩散方法的去噪估计、梯度引导、伪逆投影与数据一致性方程，以及 pMF 速度积分和 LoRA 内层更新见 [方法公式](../ct/diffusion_methods.md)。各算法的来源参数依据与使用限制保留在下方卡片。


Budgeted DPS 保留显式子集调度、成本函数和 reformulated_gradient。CT 映射限于无探测器掩膜的 Torch 二维平行束，其他任务需原生 SubsettableForwardOperator。schedule_actions 为非空的非负整数索引数组列表，key_indices 可空。本次范围为代码、接口、参数与来源资源。
