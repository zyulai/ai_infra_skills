# PR #23860: [perf] cherry-pick #22911: support return_routed_experts with overlap scheduling

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ByronHsu |
| 合并 | 2026-04-28 |
| 分类 | Overlap / Scheduler(tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/23860 |

## 一句话结论
把 routed-experts 捕获的同步 DtoH 拷贝改为随 overlap 调度的延迟拷贝,`--return-routed-experts` 下端到端输出吞吐 +15.5%。

## 动机
此前 routed-experts 捕获路径在 forward step 内做同步 device-to-host 拷贝(`out_cache_loc.cpu()` + `device_cache.buffer[...].cpu()`),迫使 GPU 排空等待,破坏 overlap scheduler,导致 `--return-routed-experts` 在 overlap 模式下不可用。本 PR 是上游 #22911(commit `c5603268`)cherry-pick 到 `sglang-miles` 分支。

## 优化点
新增 `RoutedExpertsOutput` dataclass:持有 GPU 张量与 host cache 引用,`copy_to_cpu()`(non-blocking,在 forward stream 内、`copy_done.record()` 之前调用)与 `finalize()`(`copy_done.synchronize()` 后写 host cache buffer)。`_RoutedExpertsCapturerReal` 抽出 `_get_local_range` 并新增 `_prepare_routed_experts_output`;`on_forward_end` 增加 `no_copy_to_cpu` 参数,overlap 启用时(`no_copy_to_cpu = not server_args.disable_overlap_schedule`)走延迟路径并返回 `RoutedExpertsOutput`,否则保持原同步路径。结果经 `ModelRunnerOutput.routed_experts_output` → `GenerationBatchResult` → `TpModelWorker` → `EAGLEWorkerV2`/`MultiLayerEagleWorkerV2` 一路传递,`SchedulerOutputProcessorMixin` 在 `copy_done.synchronize()` 后调用 `finalize()` 并清空。cherry-pick 时还保留了 miles 侧三处改动(draft-worker guard、spec decoding 的 cuda-graph token 数、DeepEP all-gather 跳过 DP 切片)。

## 关键改动

```text
python/sglang/srt/layers/moe/routed_experts_capturer.py: RoutedExpertsOutput + no_copy_to_cpu 延迟拷贝路径
python/sglang/srt/model_executor/model_runner.py: overlap 启用时 no_copy_to_cpu=True,输出挂到 ModelRunnerOutput
python/sglang/srt/managers/scheduler_output_processor_mixin.py: copy_done.synchronize() 后 finalize
python/sglang/srt/managers/utils.py 等: GenerationBatchResult / 两个 EAGLE worker 的字段传递
```

## 性能声明
H200 TP=4、Qwen3-30B-A3B、batch=64、in=1024、out=512,overlap 模式 + `--return-routed-experts`:Before(同步 DtoH)7453.48 tok/s → After(延迟拷贝)8609.21 tok/s,即 +15.5% 端到端输出吞吐。

## 验证与回归风险
Router replay 精度测试 `test_return_routed_experts` 3/3 通过,约 26.7M 个 expert ID 零 mismatch。风险:finalize 依赖 `copy_done.synchronize()` 的时序,prefill/decode 两个 output processor 路径都需覆盖;非 overlap 模式走原同步路径;draft worker 显式跳过,spec decoding 与 DeepEP 分支的行为差异在 cherry-pick 冲突解决中单独保留,需在对应模式下验证。

## 关联
上游原 PR #22911;同为 overlap 调度方向(同分类其他 PR 编号素材未给出)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
