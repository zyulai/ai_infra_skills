# PR #45978: [Perf] diffusion: Keep every component resident in auto mode when memory allows

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-24 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/45978 |

## 一句话结论

diffusion 推理 `--performance-mode auto` 新增内存富余档:显存装得下全部权重 2 倍时所有组件常驻,省掉 PCIe 搬运,Z-Image-Turbo e2e 快约 9%。

## 动机

auto 模式此前按静态 per-model allowlist 决定常驻组件:高于 `keep_resident_min_available_gb` 阈值(图像模型 45 GB、视频模型 120 GB)后只保留 VAE(图像模型再加 DiT),其余组件逐层 offload。于是在 288 GB 的 B300 上 Z-Image、FLUX、Qwen-Image 等仍在逐层搬运 text encoder,Z-Image-Turbo 一个 0.354 s 的请求里有 0.066 s 花在 PCIe 搬 3.4 GB;Wan2.2-A14B 更是在 H200/B200/B300 上也逐层流式两个 14B expert,并钉住约 54 GB 主机内存。本应修复此问题的 warmup-calibrated planner(#37917/#37918)未合并,本 PR 提供一个保守的过渡规则。

## 优化点

核心是给 auto 模式加一个"内存富余档"(`_fits_all_components_resident`):当最空闲的选中 GPU 空闲显存 ≥ 每 GPU 权重字节数的 2 倍时,所有无显式 placement 的组件全部常驻,不再套用默认逐层 offload 清单。权重在加载前从本地缓存 checkpoint 的 safetensors header 估算(`checkpoint_component_sizes`):按 `model_index.json` 逐组件统计参数量,以服务精度(否则按存储精度)折算字节,DiT 按 tp_size 分摊,text encoder 即使 TP-folded 也按全副本计(保守)。档位在 checkpoint 未完全缓存、含非 safetensors/量化权重、或被 `--quantization`/`--transformer-weights-path`/SVDQuant/component 覆盖时不生效;dynamic batching、共享主机/设备内存(GB10)、MPS 同样排除。规则平台无关,ROCm 上符合条件也保持常驻。另外 Wan2.2-T2V/I2V-A14B 显式设 `keep_resident_min_available_gb=120`、DiT+VAE 常驻,H200 级以上双 expert 上卡,80 GB 卡继续流式。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/server_args/auto_tune.py: 新增 _keeps_all_components_resident/_fits_all_components_resident 内存富余档判定与常驻集合替换
python/sglang/multimodal_gen/runtime/loader/utils.py: 新增 checkpoint_component_sizes,从 safetensors header 逐组件估参量与字节数
python/sglang/multimodal_gen/configs/pipeline_configs/wan.py: Wan2.2-A14B 设 keep_resident_min_available_gb=120、keep_resident_components=("dit","vae")
docs/docs/sglang-diffusion/deployment_cookbook.mdx: 更新 auto 模式说明
python/sglang/multimodal_gen/test/server/perf_baselines/h100.json 等: 更新 VRAM 基线
```

```python
+        weight_gb = weight_bytes / BYTES_PER_GB
+        fits = available_gb >= AUTO_ALL_RESIDENT_WEIGHT_MULTIPLE * weight_gb
```

## 性能声明

- Z-Image-Turbo,1xH200,flagless auto,1024x1024、9 步,2 次预热后测 5 个不同 prompt 请求,main/PR 交替两轮:e2e 中位数 0.952/0.954 s → 0.867/0.863 s(−9%),`TextEncodingStage` 0.130 s → 0.031 s,峰值显存 15,456 MB → 22,170 MB。
- H100 Z-Image-Turbo 全常驻 vs auto:0.701/0.699 s vs 0.793/0.799 s(−12%)。
- 显存被其他进程占至约 30 GiB 时 placement 与 main 一致(不回退变差)。
- Wan2.2-T2V-A14B on 1xH200(仅启动,未计时):双 expert 上卡,每 rank 少钉约 52 GB pinned 主机内存;e2e 增益未测量。
- PR 正文注明部分模型行(FLUX/Qwen-Image 等)由 sizing 规则推导,未实际运行。

## 验证与回归风险

- 新增单测:`test_server_args.py`(各显存档位/量化/批处理/off-CUDA 下 placement)、`test_weight_utils.py::TestCheckpointComponentSizes`(本地布局与缓存 hub snapshot,部分下载/缺 shard/.bin/FP8 均判 unsized),PR 称这些测试在 main 上会失败。
- 精度:纯 placement 变化,同模块同设备,H200 上 5 个 prompt 输出 md5 与 main 一致。
- 风险:常驻后稳态 VRAM 上升(如 Z-Image-Turbo +6.7 GB),H100 等多条 CI 显存 up-only 检查会按设计失败,需按 PR 列表重新 seed 基线;首次启动 checkpoint 未缓存时 placement 与 main 相同;`ServerArgs` 启动时新增读取本地 safetensors header 的开销。CI 状态显示 Extra 与 AMD ROCm 10 两条为失败。

## 关联

- 依赖/前序:#37917、#37918(warmup-calibrated planner,未合并,本 PR 为过渡方案)。
- 同区域:diffusion 多模态 placement/auto 内存策略系列 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
