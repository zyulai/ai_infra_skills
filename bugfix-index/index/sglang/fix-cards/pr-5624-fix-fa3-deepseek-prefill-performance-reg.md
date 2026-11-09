# PR #5624: Fix FA3 DeepSeek prefill performance regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5624](https://github.com/sgl-project/sglang/pull/5624) |
| 作者 | Alcanderian |
| 合并日期 | 2025-04-22 |
| labels | (none) |
| 热度 | 👍 8 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
FA3 + DeepSeek 在无前缀的 extend prefill 上误选 MLA 路径造成性能回退;修复为 sum(prefix_lens)==0 时也走 MHA_CHUNKED_KV,input-len 1024 prefill 提速约 20%。

## 问题现象
FA3(Flash Attention 3)后端跑 DeepSeek 时 prefill 性能回退。作者用 `python3 -m sglang.bench_one_batch --model lmsys/sglang-ci-dsv3-test --batch-size 8 --input-len 1024 --output-len 128 --trust-remote-code` 对比:修复前 Prefill 0.32884 s / 24911.50 token/s,修复后 0.27612 s / 29668.45 token/s,即 context len 1024 时提升约 20%;正文称「the longer context, the more benifitions」(上下文越长收益越大)。decode 吞吐基本不变(约 560 token/s)。

## 根因
正文未披露根因(正文只写了动机与基准数据,无 Root Cause 一节)。(由 diff 推断)fa3 后端的 `dispatch_attn_forward_method` 此前只有当 `sum(extend_prefix_lens_cpu) >= chunked_prefix_cache_threshold` 时才选 `MHA_CHUNKED_KV`,于是没有任何前缀的普通 extend(prefill)会落入 MLA 分支;而 FA3 的 MLA 路径在 prefill 上比 MHA chunked KV 慢,表现为上述性能回退。正文另附进一步建议:选择 MLA 或 MHA CHUNKED KV 应同时考虑 context len 与 prefix len(两者都影响核心 attention 的 FLOPS),推导出 `prefix < C * (context^2 + prefix*context)`(C=0.01171875)的过滤条件,并建议实现 `sum(prefix) < C * sum(context^2 + prefix*context)` 的过滤器——该建议不在本次 diff 内。

## 修复方案
修改 `python/sglang/srt/models/deepseek_v2.py` 的 `dispatch_attn_forward_method`(fa3 分支):当 `extend_prefix_lens_cpu` 非 None 时计算 `sum_extend_prefix_lens`,选 `MHA_CHUNKED_KV` 的条件从 `sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold` 放宽为「超过阈值 **或** `sum_extend_prefix_lens == 0`」,即无前缀的 prefill 也走 MHA_CHUNKED_KV:

```python
+            if forward_batch.extend_prefix_lens_cpu is not None:
+                sum_extend_prefix_lens = sum(forward_batch.extend_prefix_lens_cpu)
             if (
                 forward_batch.forward_mode.is_extend()
                 and not self.disable_chunked_prefix_cache
                 and not forward_batch.forward_mode.is_target_verify()
                 and not forward_batch.forward_mode.is_draft_extend()
                 and (
                     sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold
+                    or sum_extend_prefix_lens == 0
                 )
             ):
                 return AttnForwardMethod.MHA_CHUNKED_KV
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响 fa3 后端 + DeepSeek(V2 系)模型的 prefill 分发逻辑:所有无前缀(prefix lens 全 0)的 extend 都会从 MLA 改走 MHA_CHUNKED_KV;有前缀且超阈值、target_verify、draft_extend、disable_chunked_prefix_cache 等路径行为不变。

## 关联
正文未列出 issue/PR 链接(仅 @ 了 Fridge003 征求意见)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
