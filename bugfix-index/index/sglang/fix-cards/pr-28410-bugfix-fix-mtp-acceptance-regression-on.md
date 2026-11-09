# PR #28410: [Bugfix] Fix MTP acceptance regression on plan stream by moving int64 cast before plan stream context

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28410](https://github.com/sgl-project/sglang/pull/28410) |
| 作者 | iridiumine |
| 合并日期 | 2026-06-21 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 1 · 💬 12 |
| 分类 | Speculative Decoding |
| bug 类型 | regression |

## 一句话结论
PR #28043 引入的 int64 cast 在 plan stream 内执行,与主 stream 产生数据竞争损坏 draft token ids,MTP 接受率趋近零;将 cast 移到进入 plan stream 之前。

## 问题现象
开启 overlap plan stream(环境变量 `ASCEND_USE_FIA=1`、`SGLANG_ENABLE_SPEC_V2=1`、`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`,NPU/ascend 后端,NextN/MTP 投机解码)时,MTP 接受率几乎降为 0。正文未给出具体报错日志;修复后用 Qwen3.6-35B-A3B 跑 gsm8k mean_acc 恢复到 0.96。

## 根因
正文披露:PR #28043 为修复 DeepSeek-V4 DP dtype mismatch hang,在 `prepare_for_draft_extend` 内把 predict 归一化为 `torch.int64`;但当 overlap plan stream 开启时,这个 cast 会在 plan stream 上执行,而 predict 是在主 stream 上产生的,形成跨流数据竞争,悄悄损坏 draft token ids,导致 MTP 接受率掉到接近零。

## 修复方案
- `python/sglang/srt/speculative/eagle_worker_v2.py` `_draft_extend_for_decode`:在进入 `plan_stream_ctx` 之前先执行 `next_token_ids = batch_result.next_token_ids.to(torch.int64)`,再把已 cast 的张量传入 `prepare_for_draft_extend`,保证 cast 在主 stream 上完成、有正确同步。
- `python/sglang/srt/speculative/base_spec_worker.py` `prepare_for_draft_extend`:移除函数内的 `predict.to(torch.int64)`,直接 `batch.input_ids = predict`,并更新注释说明为何 dtype 归一化不能放在该函数内、调用方须在进入 plan stream 前 cast。

```diff
+        next_token_ids = batch_result.next_token_ids.to(torch.int64)
         with self.plan_stream_ctx:
             forward_batch = self.prepare_for_draft_extend(
                 draft_extend_input,
                 batch,
-                batch_result.next_token_ids,
+                next_token_ids,
```

## 影响与回归风险
- 影响窗口:自 PR #28043 引入该 cast 起、至本修复合并前,且仅在 overlap plan stream 开启时触发。
- 回归面:投机解码 v2 的 draft extend 路径(base_spec_worker 的 `prepare_for_draft_extend` 与 eagle_worker_v2 的 `_draft_extend_for_decode`)。仅改变 cast 的执行位置,不改变 token id 数值,对模型输出无影响;DeepSeekV4 DP gather 需要各 rank input_ids dtype 一致的约束仍由调用方的提前 cast 满足。若未来新增调用 `prepare_for_draft_extend` 的路径,需遵守「进入 plan stream 前 cast」的约定(注释已标明)。

## 关联
- https://github.com/sgl-project/sglang/issues/28405
- https://github.com/sgl-project/sglang/issues/28406
- https://github.com/sgl-project/sglang/pull/28043

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
