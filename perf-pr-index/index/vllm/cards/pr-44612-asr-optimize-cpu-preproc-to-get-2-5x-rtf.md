# PR #44612: [ASR] Optimize CPU preproc to get 2.5x RTFx via multi-threading

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ekagra-ranjan |
| 合并 | 2026-06-12 |
| 分类 | Overlap / Scheduler \| tier: major |
| Label | performance, frontend, ready, multi-modality, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/44612 |

## 一句话结论

ASR CPU 预处理改用 ThreadPool，长音频 RTFx 735→1890（2 threads，2.5x）。

## 动机

正文指出 vLLM ASR 的 CPU 预处理（audio loading、RMS chunking）是同步代码，会阻塞 FastAPI server 的主 eventloop，导致串行执行而非批处理；处理长音频（#44587）时这一瓶颈更严重。高并发下还会让 `health/` 端点超时，使故障节点难以识别、自动扩缩容过于激进。

## 优化点

把这些同步 CPU 预处理 offload 到 ThreadPool executor，让 asyncio 主 eventloop 在后台线程加载音频数组、做 RMS chunk split 的同时继续运行 server。新增环境变量 `VLLM_MAX_AUDIO_PREPROCESS_WORKERS` 控制线程数。正文未展开线程池的实现细节，只说明卸载同步预处理这一机制。

## 关键改动

```text
vllm/entrypoints/speech_to_text/base/serving.py: ASR 服务的 CPU 预处理改为 offload 到 ThreadPool
vllm/utils/async_utils.py: 提供异步卸载到线程池的工具
vllm/envs.py: 新增 VLLM_MAX_AUDIO_PREPROCESS_WORKERS 环境变量
vllm/entrypoints/serve/utils/server_utils.py: 服务侧相关改动
```

## 性能声明

正文给出 "2.5x higher RTFx throughput for `openai/whisper-large-v3-turbo`"，并称 `health/` 端点延迟从 25s 降到 0.06ms。1xH100 上长音频（ArtificialAnalysis/Earnings22-Cleaned-AA，MAX_CONCURRENCY=128、NUM_PROMPTS=128）逐档数据：w/o this PR RTFx 735、Benchmark duration 201.75；this PR 2 threads RTFx 1890、duration 78.45（"250% gain with 2 thread"），1 thread 1335.05、4 thread 1739.89、8 thread 1272.91、16 thread 1292.97。短音频（>80% 样本 <10s，MAX_CONCURRENCY=500）w/o PR RTFx 600.885、this PR 591.7725（1.5% drop）；若以 `VLLM_MAX_AUDIO_PREPROCESS_WORKERS=1` 启动则得到 1.7% higher RTFx。硬件 1xH100，模型 openai/whisper-large-v3-turbo，精度 PR 未注明。

## 验证与回归风险

正文说明 `test_long_audio_wer_correctness`（由 #44587 引入）通过，并将在 CI 上 gate 该代码路径；该测试覆盖长音频 + BS>1 + multi-threading + RMS chunk，数据集仅 37MB 故适合 CI。回归风险：短音频场景有 1.5% 吞吐下降，可用单 worker 换回 +1.7%；线程数继续增大（4/8/16 threads）RTFx 反而低于 2 threads。PR 未注明对其他模型或硬件的回归评估。

## 关联

正文指明 benchmark 需在 PR #44587 合并后运行，且 gate 测试 `test_long_audio_wer_correctness` 由 #44587 引入。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
