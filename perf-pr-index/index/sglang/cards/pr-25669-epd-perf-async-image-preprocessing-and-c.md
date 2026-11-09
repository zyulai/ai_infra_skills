# PR #25669: [EPD][Perf] Async image preprocessing and cross-request ViT batching for encode_server

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ntgiang71096 |
| 合并 | 2026-06-01 |
| 分类 | Overlap / Scheduler |
| Label | run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/25669 |

## 一句话结论

把 EPD encoder 的多模态预处理移出 asyncio 事件循环并发执行,MI355X 上 E2E 延迟 -18%、吞吐 +26%。

## 动机

EPD encoder server(encode_server.py)的图像预处理(PIL resize + normalize,经 image_processor())在 asyncio 事件循环上同步执行:N 个并发请求的预处理被完全串行化,每个请求都要等前一个做完;这也导致并发请求到达时批次化 ViT 前向无从谈起。原计划同时做 cross-request ViT batching,评审后按 #25964 的分工收窄(见下)。

## 优化点

新增专用 `preproc_executor`(ThreadPoolExecutor),把 image_processor / video_processor / audio_processor.feature_extractor 调用经 `run_in_executor` 移出事件循环,N 个并发请求可并行预处理,不再阻塞请求接收与彼此;与处理 ZMQ 发送的 `self.executor` 分开,避免高并发下争用。worker 数由 `SGLANG_ENCODER_PREPROC_WORKERS` 控制(评审后默认从 32 降到 8)。注意范围变化:初版还包含 `_vit_batch_loop` 跨请求 ViT 批处理(队列 + 自适应 sleep),评审中被整体移除——该能力由 PR #25964 在更粗一层(EncoderScheduler + batch_encode,连 HF processor 调用一起融合)覆盖;两个改动互补,本 PR 让事件循环不被阻塞,#25964 的 batch worker 才能看到真正并发的请求。作者重测确认收窄后性能不变。

## 关键改动

```text
python/sglang/srt/disaggregation/encode_server.py: 新增 preproc_executor,image/video/audio 预处理 run_in_executor 化
python/sglang/srt/environ.py: SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
```

## 性能声明

AMD MI355X、Qwen/Qwen3.6-35B-A3B、EPD 1E1P;bench_serving,8 张 720p 图、100 prompts、random-input-len 128、output_len=1(隔离 encoder 路径:预处理 + ViT + ZMQ + PD prefill,无 decode 压力)、max-concurrency 16。Stock 5820 ms 中位 E2E / 2.59 req/s;收窄后(仅 async 预处理)重测为 -18% 延迟 / +26% 吞吐;初版表格(含 ViT batching)为 4705 ms / 3.19 req/s(-19% / +23%)。实测时 `SGLANG_ENCODER_BATCH_SIZE=16`,preproc workers 有效并行度受 max-concurrency 16 封顶,≥16 即可复现。

## 验证与回归风险

初版在 test/registered/unit/disaggregation/test_encode_server_perf.py 加了 5 个测试(async 并发与 ViT 批队列),评审后认为其只测 mock 上的 asyncio/ThreadPoolExecutor、保护价值不大而删除;最终验证为 pre-commit 通过、CI base/extra 通过与上述 benchmark。准确性方面作者声明无影响——预处理与 ViT 推理本身不变,只改调度时机。回归面:run_in_executor 包裹 video/audio 路径同样生效;线程池默认 8 worker,极端并发下可能成为新的排队点(可通过 env var 调大);平台无关(标准 asyncio + concurrent.futures)。

## 关联

后续/互补 PR #25964:在更粗层实现 cross-request ViT batching(EncoderScheduler + batch_encode),接管本 PR 移除的 ViT 批处理部分。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
