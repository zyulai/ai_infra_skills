# PR #31622: Revert "Fix mamba track-boundary seqlen under overlap scheduler (#31369)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31622](https://github.com/sgl-project/sglang/pull/31622) |
| 作者 | alisonshao |
| 合并日期 | 2026-07-18 |
| labels | (none) |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论

#31369 改用错位的 track seqlen 造成 Qwen3-Next decode-cache-hit 确定性精度回归;revert 恢复 `kv_committed_len` 判界,正确修法走 #29792。

## 问题现象

`main` 上出现确定性正确性回归:Qwen3-Next 的 decode-cache-hit logprobs 与冷启动重算(cold recompute)不一致,nightly `4-gpu-h100` 测试自 2026-07-17 起失败(照抄原文):

```
test/registered/models_e2e/test_qwen3_next_models.py::test_input_output_logprobs_match_decode_cache_hit_helper
AssertionError: avg_kl_div=0.006489 > threshold=0.002 for Qwen/Qwen3-Next-80B-A3B-Instruct
```

每次重试 kl_div 完全相同,说明是确定性问题而非 flaky。

## 根因

正文披露(含 bisect 证据):回归被定位到 #31369(`0675d3033f`)——kl_div 在 `dc0b3eb` 时 ≤0.0005(通过),到 `681c2235` 跳到 0.0065(失败),#31369 在好 run 中不存在、坏 run 中存在,且是该窗口内唯一改动 `_mamba_check_track_boundary`/`batch_result_processor.py` 的提交。机制上,#31369 把 boundary seqlen 从 `req.kv_committed_len` 改为 `len(req.origin_input_ids) + len(req.output_ids) - 1`,新的 seqlen 在 decode-cache-hit 场景下给 mamba track boundary 贴错标签,使被复用/捐赠的 tracked prefix 从一个冷 prefill 重算不会产生的状态继续延伸,正是该测试捕捉的不一致(也是 #31369 自己描述里警告过的症状)。

## 修复方案

revert #31369,恢复 `batch_result_processor.py` 中 `_mamba_check_track_boundary` 的非 spec 分支为按 `kv_committed_len` 判界:

```diff
         if batch.spec_algorithm.is_none():
-            seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
-            if seq_len % interval == 0:
-                return True, seq_len
+            if req.kv_committed_len % interval == 0:
+                return True, req.kv_committed_len
```

恢复后边界条件与 forward 的 tracking mask 一致(`prepare_for_decode` 同步递增 `seq_lens_cpu` 与 `kv_committed_len`,值始终为 interval 的倍数、页对齐)。

## 影响与回归风险

- 影响窗口:自 #31369 合并起至本 revert(2026-07-17 起的 nightly 受影响)。
- 回归面:仅 mamba 模型在 overlap scheduler 下的 track-boundary 记账(`_mamba_check_track_boundary` 非 spec 分支);revert 恢复先前通过的行为,#31369 原本针对的 overlap 边界问题重新存在,需经开放中的 #29792(以正确的记账方式针对同一代码)重新落地。作者说明 4×H100 devbox 上的硬件 A/B 确认(revert 后 kl_div 回到阈值内)当时仍在进行中。

## 关联

- #31369(被 revert 的 PR)
- #29792(承接原问题的正确修法)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
