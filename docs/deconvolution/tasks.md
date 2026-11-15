# Deconvolution tasks

Primary showcase/quality inputs: [standard public-image benchmarks](../public_benchmarks.md)
with Levin09 and ideal Airy PSFs. Run tools/run_public_benchmarks.py.
The algorithm/input mapping is in [public benchmarks](../public_benchmarks.md#deconvolution).

DeconvolutionTaskRequest and run_deconvolution_task accept external arrays/files.
wiener_hunt and richardson_lucy execute reconstruction without a catalog or truth.
TV ADMM, RL-TV and Hyper-Laplacian also execute. blind_dcp and blind_l0 execute task_type=blind_deconvolution; see ../public_benchmarks.md.
No algorithm substitution, search or retry occurs. See algorithms.md,
data_format.md and ../public_benchmarks.md for executable calls.

See ../public_benchmarks.md for external-input API/CLI examples and editable benchmarks.

## 聚合能力入口

`inv_framework.capabilities.list_capabilities("deconvolution", context=context)` 将上下文传入本域能力查询。领域请求、默认参数和结果遵循本手册，文件任务与独立调用共享 runner 契约。公共安装和 harness 绑定见 [任务 API](../agent_algorithm_api.md)与[Harness](../ct/harness.md)。
