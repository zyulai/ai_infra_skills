# PR #24195: Fix flashinfer autotune oom glm51

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24195](https://github.com/sgl-project/sglang/pull/24195) |
| 作者 | kpham-sgl |
| 合并日期 | 2026-06-03 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
FlashInfer autotune 的 dummy run 仍跑 LM head + all-gather,DP attention 下 OOM;autotune 期间返回空 logits 跳过修复。

## 问题现象
标题指明 flashinfer autotune 在 glm51 上 OOM。正文未描述具体现象与报错日志,仅说明 "Brought over from PR #23796 to trigger Nightly CI"(从 #23796 带过来以触发 Nightly CI)。

## 根因
正文未披露根因。

(由 diff 推断,摘自 diff 注释)FlashInfer autotune 只 profile attention/MoE/GEMM kernel,但 autotune 的 dummy run 仍会完整执行 LM head 与 tensor-parallel all-gather;其 `[batch * dp_size, vocab]` 的 logits 输出在 DP attention 且 `mem_fraction_static` 紧张时直接 OOM,而这份输出本会被丢弃,all-gather 也是无效功。

## 修复方案
1. `python/sglang/srt/layers/logits_processor.py`:新增模块级开关 `_in_autotune_dummy_run`、`get_in_autotune_dummy_run()` 与 context manager `autotune_dummy_run_mode()`;`forward` 在 MIS / DLLM / common 三个 dispatch 之前检查开关,命中则直接返回空输出,跳过全部 LM head 路径与 all-gather:

```diff
+        if _in_autotune_dummy_run:
+            return LogitsProcessorOutput(next_token_logits=None)
```

2. `python/sglang/srt/model_executor/model_runner.py` 的 `_flashinfer_autotune`:dummy run 外再套一层 `autotune_dummy_run_mode()`:

```diff
-            with torch.inference_mode(), autotune(True, cache=str(autotune_cache)):
+            with (
+                torch.inference_mode(),
+                autotune(True, cache=str(autotune_cache)),
+                autotune_dummy_run_mode(),
+            ):
                 self._dummy_run(batch_size=self.req_to_token_pool.size)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 FlashInfer autotune 的 dummy run 路径——开启后 dummy run 的 logits 输出为空(`next_token_logits=None`),不再分配 LM head all-gather 输出;正常推理 forward 不受影响(全局开关默认 False,仅在 autotune 上下文内置位)。若存在依赖 dummy run logits 输出的逻辑,会拿到 None,但 autotune 只 profile kernel、不消费该输出。

## 关联
- 来源 PR:sgl-project/sglang#23796(Brought over from)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
