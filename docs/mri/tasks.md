# MRI tasks

Primary showcase/quality inputs: [standard lightweight benchmarks](../public_benchmarks.md).
Run tools/run_public_benchmarks.py; the analytic MR phantom is separate from
the authorized local fastMRI slice needed for clinical-distribution VarNet evaluation.

Use MRITaskRequest and run_mri_task from inv_framework.mri.api, or run_experiment
and inv-run. zero_filled and cg_sense accept external observations, without a
catalog or reference image. GRAPPA, CS, ESPIRiT calibration and checkpoint-backed
VarNet also execute. VarNet supports CPU/CUDA; other MRI methods use CPU.
Non-Cartesian sampling needs `.[mri]`; VarNet needs an explicit matching checkpoint.
See [dependencies and installation](../../DEPENDENCIES.md).

See data_format.md and algorithms.md for mathematical conventions and shapes.
For runnable generated examples see ../public_benchmarks.md. Missing sensitivity
maps returns invalid_parameters; it never triggers another algorithm.

See ../public_benchmarks.md for external-input API/CLI examples and editable benchmarks.

## 聚合能力入口

`inv_framework.capabilities.list_capabilities("mri", context=context)` 将上下文传入本域能力查询。领域请求、默认参数和结果遵循本手册，文件任务与独立调用共享 runner 契约。公共安装和 harness 绑定见 [任务 API](../agent_algorithm_api.md)与[Harness](../ct/harness.md)。
