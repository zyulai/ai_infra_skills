# PR #45977: [Perf] diffusion: Turn on breakable CUDA graphs by default for LTX-2/LTX-2.3 two-stage serving

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-24 |
| 分类 | CUDA Graph / Compiler |
| Label | documentation, lora, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/45977 |

## 一句话结论

LTX-2/2.3 two-stage serving 默认开启 breakable CUDA graphs,2xB300 端到端 −44%、2xH200 −20%。

## 动机

LTX-2 two-stage 的 stage 1 只跑半分辨率(约 1.5k video tokens),step 是 launch-bound:4xB300 上 TRT-LLM 的 CUDA-graph 配置单步 0.044 s,sglang eager 0.192 s。breakable CUDA graphs(BCG)能关闭大部分差距,且 replay 同样的 kernel、输出 bitwise 等于 eager,但此前是 opt-in,flagless 部署用不上。其他模型无收益(FLUX.3 在 10 格中只赢 1 格、Qwen-Image-2.1 单卡 −0.5%、MiniMax-H3 Ref2VA 无加速),故其余场景保持 opt-in。

## 优化点

机制三层:
1. `ModelDeploymentConfig.bcg_default_pipeline_class_names` 新字段,`LTX2PipelineConfig` 声明 `LTX2TwoStagePipeline`(LTX-2.3 继承),只对实测更快的 pipeline 类默认开启。
2. `enable_breakable_cuda_graph` 默认值 False→None;`_breakable_cuda_graph_default()` 在 unset 时按 gate 求值:opted-in pipeline + validated checkpoint、performance-mode auto/speed、CUDA 平台且非共享 host memory(GB10 关)、server warmup、monolithic 角色、非 ComfyUI、无 torch.compile/Cache-DiT/动态 batching/`--lora-path`、两个 stage DiT resident 且显存 ≥130 GiB(LTX-2.3 需 resident 模式);显式 true/false 恒优先,默认开启时打印提示日志。
3. 配套修正:video BCG warmup 改为捕获模型完整默认分辨率(此前受 server warmup 832x480 上限约束,Wan/MiniMax-H3/LTX-2.5 会捕获无人请求的 shape);LoRA 变更(set_lora/merge/unmerge)与 `resume_memory_occupation` 后 drop 已捕获 graphs 并告警,后续请求跑 eager——graph replay 捕获时的权重与 LoRA 分支,serving 从不重捕获。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/server_args/server_args.py: enable_breakable_cuda_graph 的 None 语义与 _breakable_cuda_graph_default() gates
python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2.py、model_deployment_config.py: bcg_default_pipeline_class_names 声明
python/sglang/multimodal_gen/runtime/managers/gpu_worker.py、stages/denoising.py: drop_breakable_cuda_graphs(LoRA/权重变更后)
python/sglang/multimodal_gen/runtime/warmup_request_builder.py: BCG 时 warmup 使用完整默认视频分辨率
```

## 性能声明

2 GPU CFG parallel,768x512x121:
- 2xB300,LTX-2 T2V 40 steps:eager 6.693/6.882 s → BCG 3.797/3.785 s(−44%);LTX-2.3 30 steps:6.294/6.202 → 4.735/4.726(−24%)。
- 2xH200,LTX-2 40 steps(2 次实测请求取 median,main/PR 交替 rounds):7.398/7.593 → 5.969/5.970(−20%);stage 分解:`LTX2AVDenoisingStage` 5.368→3.702 s、`LTX2RefinementStage` 1.187→1.096 s,decode 不变(0.67 s)。Hopper 上 FA attention 仍 eager,每 forward 289 段;B300 上 LTX attention 也被捕获(1 段),故增益约两倍。
- 代价:H200 启动 +6 s(缓存热;冷启约 +20 s)、峰值显存 +6.9 GB;B300 +2 s、+0.95 GB(LTX-2)/+1.1 GB(LTX-2.3)。
- 精度:seed 42 下 BCG 与 eager 各臂 MP4 字节及解码帧 md5 完全一致。

## 验证与回归风险

测试:`test_server_args.py::TestBreakableCudaGraphDefault`(mock 141 GiB CUDA GPU 上 flagless 解析为 on,显式 off/offline/request warmup/memory、manual mode/compile/batching/`--lora-path`/ComfyUI/单 stage/HQ pipeline/未验证 checkpoint/各种 offload/低显存/80 GB 卡均解析为 off,ROCm 与共享内存平台关)、`test_cfg_parallel_warmup.py::test_bcg_capture_keeps_the_full_default_video_resolution`、`test_diffusion_bcg_padding.py::TestBCGWeightChangeAfterCapture`(LoRA 变更与 resume 后 drop graphs 转 eager)。风险:只有捕获 shape(768x512x121、默认 quality、T2V)replay,其他分辨率/帧数(OpenAI video API 默认 4 秒即 89 帧)/image-conditioned/`quality="exact"` 请求回落 eager(一次性 MISSED 告警);80 GB 卡受 130 GiB gate 保护保持 eager,H100 suite 不受影响;与 #45943 的 dynamic-LoRA linear path(stage 2)交互需两者都落地后复测速度。

## 关联

正文提及 #45943(dynamic-LoRA linear path,合并后需复测);既有 opt-in BCG CI 测试 `test_diffusion_bcg_*zimage_turbo.py` 为 image 模型路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
