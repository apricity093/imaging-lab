# NaSn 独立 CT 应用

应用位于 `inv_framework/applications/nasn_ct`。固定来源为 peterw634/inv_framework 的 dc181fb65afbf300a9935c3c770231c44ffc0c78，其中 NaSn 来源提交为 35bb7088273dddb07495cafbf9fcc612c07ae930。代码许可保留在 LICENSES/NaSn-BSD-3-Clause.txt，pMF 模型及发布资源沿用各自许可。

## 安装与资源

```powershell
pip install -e ".[nasn,nasn-train]"
inv-nasn-ct doctor
inv-nasn-ct --help
inv-nasn-ct-train --help
python tools/download_deep_weights.py pmf --output D:/data/nasn/weights
inv-nasn-ct data fetch --output D:/data/nasn/downloads --extract
```

源码可使用 `python -m inv_framework.applications.nasn_ct.cli` 和 `python -m inv_framework.applications.nasn_ct.training.train`。基础 CPU 安装支持原有任务，NaSn 依赖在选定工作流时加载。CUDA 原生投影和 Gridrec 首次使用需兼容 PyTorch 的 Toolkit、C++ 编译器与 Ninja。真实 2048 模型需要足够 GPU 显存。

数据清单 `manifests/data-v0.1.0.json` 保留 `published=false`，包含 quickstart、4个 target 和4个 training 扫描档案。下载工具在未发布时返回退出码2，须提供已有本地数据或更新后的授权清单。`data verify --directory PATH` 检查已获取档案。清单中的许可为来源声明。

模型清单 `manifests/model-v0.1.0.json` 与共享模型资源声明一致，发布模型为 epoch449 EMA、pmfDiT_B_128、2048²单通道，约526 MB，权重旁需同名 JSON。模型获取复用 `tools/download_deep_weights.py`，求解时读取本地文件。预训练训练起点为外部上游 pMF-B/16 的原生 state_dict，通过 `--pretrained` 指定，来源清单没有提供该起点的固定下载地址。

## 数据处理与尺度

```powershell
inv-nasn-ct data prepare-target --raw-root D:/data/nasn/raw/target/scan1 --output D:/data/nasn/target/scan1 --device cpu
inv-nasn-ct data calibrate-target --sinogram-dir D:/data/nasn/target/scan1/sinograms_rivers --output D:/data/nasn/target/scan1/scale_stats_center1172.json --center 1172
inv-nasn-ct data prepare-fullct --config inv_framework/applications/nasn_ct/configs/fullct_scan1.yaml --raw-root D:/data/nasn/raw/training/scan1 --output D:/data/nasn/fullct/scan1
inv-nasn-ct data build-corpus --train D:/data/nasn/fullct/scan1/recon_gridrec D:/data/nasn/fullct/scan2/recon_gridrec D:/data/nasn/fullct/scan3/recon_gridrec --val D:/data/nasn/fullct/scan4/recon_gridrec --output D:/data/nasn/corpus
```

FullCT 对 scan2–4 使用对应配置重复运行。安装后可通过 Python `Path(inv_framework.applications.nasn_ct.__file__).parent / 'configs'` 定位配置。目标预处理默认配置随包解析。小范围使用 `--slice-start`、`--slice-stop`，均为闭区间、1-based 行号。`prepare-fullct --stage sinograms` 只执行预处理。Gridrec 与目标标定使用 CUDA。

预处理保留平暗场校正、PAD-PA 相位恢复、Rivers 去条纹与 Gridrec。投影临时 memmap 保留在输出的 `.nasn-work`，恢复检查输入路径和配置。更换输入内容须选择新的输出目录。训练集构建使用新目录，避免残留旧切片。训练扫描1–3各1001张，验证扫描4共1001张，标定统计仅取训练扫描。FullCT 图像经百分位尺度乘法后使用 OpenCV CUBIC 缩放到2048²，目标投影使用 AREA 沿探测器轴2560→410，两侧各补819列。

目标中心按像素中心映射为 `819+(1172+0.5)*410/2560-0.5=1006.283203125`。尺度文件必须标明原中心1172和正的 final_scale。模型域为 `x_model = final_scale * x_phys`，保持零偏置。公开任务传入 `image_scale=1/final_scale`，输出 reconstruction.npy 为物理域，额外 TIFF 使用训练尺度。显式410列掩膜同时作用于正投影与伴随。

## 训练、恢复与导出

```powershell
inv-nasn-ct-train --data-root D:/data/nasn/corpus --pretrained D:/data/nasn/upstream-pmf-B-16.pt --workdir D:/data/nasn/train --batch-size 16 --epochs 500
inv-nasn-ct-train --data-root D:/data/nasn/corpus --resume D:/data/nasn/train/ckpt_epoch_449.pt --workdir D:/data/nasn/train --batch-size 16 --epochs 500
inv-nasn-ct weights export --checkpoint D:/data/nasn/train/ckpt_epoch_449.pt --output D:/data/nasn/weights/nasn-pmf-2048-epoch449-ema.safetensors
```

多卡使用 `torchrun --nproc_per_node=4 -m inv_framework.applications.nasn_ct.training.train` 加同一组参数，全局 batch 必须可整除进程数。默认架构和损失读取 `configs/training_2048.json`，发布配方包括 AdamW lr=4e-4、betas=(0.9,0.95)、10 epoch warmup/cosine、EMA=0.9999、LPIPS VGG权重0.4、500 epochs，增强默认关闭。LPIPS 使用其外部 VGG 权重缓存，离线运行需预置该缓存。`--model-config` 接受 pixelMeanFlow 构造参数 JSON 覆盖，恢复时须保持模型配置一致。`--device cpu` 可用于小规模流程检查，`--from-scratch` 显式选择随机初始化。训练按 epoch 保存模型、EMA、优化器和调度器及模型配置，并生成 training.json。恢复继续 epoch 边界的参数和优化状态，随机流从命令种子重新初始化。

EMA 导出过滤 LPIPS、训练专用 v-head 及确定性复数 rope buffer，写出 safetensors 和同名 JSON，加载复用共享 `solvers.deep.models.weights.load_pmf`。checkpoint 和EMA元数据保存实际 noise_scale，共享加载器按该值构造模型。旧配置缺省或显式为null时使用对应架构的来源默认值，例如B_128为8。短训练与完整500-epoch训练的结果分别记录。

## 重建与结果

```powershell
inv-nasn-ct reconstruct --weights D:/data/nasn/weights/nasn-pmf-2048-epoch449-ema.safetensors --sinogram-dir D:/data/nasn/target/scan1/sinograms_rivers --scale-stats D:/data/nasn/target/scan1/scale_stats_center1172.json --slices 1150 --num-steps 16 --dc-lam 0.05 --dc-iter 10 --device cuda --output outputs/nasn_real
```

单切片由 `slice_experiment` 构造公开 `pmf_dds` CT 请求，使用 runner 和既有 NaSn 后端、pMF、DDS、严格伴随 CG。默认seed=0、eta=0，按样本独立CPU随机数流。`--model-config` 可指定相邻配置以外的 JSON，公开参数使用同名 model_config。`--warm-start --t-start 0.5` 使用底层 Gridrec 初值。公开 pmf_dds 支持1–20步，warm_start 为真时 t_start 在(0,1)，冷启动为1。

每个切片输出目录包含 result.json、物理域 reconstruction.npy 和执行诊断，顶层包含 summary.json 和训练尺度 recon_slice_*.tif。执行失败也保存结构化结果，并使 CLI 失败退出。真实数据无配对参考，仅报告观测残差和实际耗时。

合成评估保留独立 `benchmark --weights FILE --val-root PATH --steps 16 32 --output outputs/nasn_benchmark`。该来源协议使用2048角度、512有效探测器、CG λ=5、CG前(x+1)/2与CG后2x−1，ROI半径256的圆内PSNR和512²中心方框SSIM，动态范围取参考全图max−min。16/32步来源门槛分别为35dB/0.80和37dB/0.84，该低层32步协议独立保留。输出 progress.json 和 summary.json。

## 验证范围与来源记录

来源2026-10-28记录真实切片1150迁移图像相对L2=0.000668712、测量残差0.02729678899与0.02729744464，通过其预设真实对照门槛。来源合成切片991跨运行得到36.6072/0.81288与36.2930/0.80194，严格跨运行对照失败。来源记录 CUDA BP 原子累加的波动可能被CG放大，回放算子响应时内部输入逐值一致。真实扫描的数据可用性沿用来源清单，合成数据流程不代表真实扫描效果。

文件请求模板位于 configs/ct/nasn_reconstruction.yaml，包含完整900角度和410列掩膜。使用前替换权重和观测路径，设置 image_scale=1/final_scale。观测 NPY 需为已AREA缩放和补零的物理域(1,1,900,2048)数组。
