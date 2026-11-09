# PR #8677: Fix accuracy drop of dsv3 run in dp enablement

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8677](https://github.com/sgl-project/sglang/pull/8677) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-09-04 |
| labels | high priority, ready-to-merge |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
dsv3 在 aiter 后端开启 DP attention 后精度下降,原因是 extend 阶段走了 MHA 注意力;改为用 absorb(MLA)注意力处理 extend,GSM8K 恢复到 0.937。

## 问题现象
dsv3(DeepSeek-V3)模型在 aiter 后端开启 DP(dp enablement)运行时出现精度问题(Motivation: "Got an accuracy issue of dsv3 model run on aiter backend with dp enablement"),关联 issue #7692。正文未附错误数值,仅以 GSM8K 修复后结果佐证。

## 根因
正文披露的根因方向:extend attention 此前使用 MHA attention 处理,是不正确的;Modifications 写明 "For extend attention, use absorb attention to handle it, not use the MHA attention."(extend 应使用 absorb/MLA 注意力而非 MHA)。

(由 diff 推断)具体机制:`aiter_backend.py` 的 `forward_extend` 原先在 extend 分支里把 KV 前缀与 extend 部分拼接后统一走 `flash_attn_varlen_func`(MHA 形式),该路径在 DP attention 开启时不正确;`deepseek_v2.py` 的 `_dispatch_mla_subtype` 在满足条件时无条件返回 `AttnForwardMethod.MHA`,导致开启 DP attention 的 extend 请求被派发到 MHA 路径。

## 修复方案
改两个文件:

1. `python/sglang/srt/models/deepseek_v2.py` `_dispatch_mla_subtype`:原先该分支无条件 `return AttnForwardMethod.MHA`;现当 `is_dp_attention_enabled()` 时,若 `sum(forward_batch.extend_prefix_lens_cpu) == 0`(无前缀)仍走 MHA,否则返回 `AttnForwardMethod.MLA`(absorb);未开 DP attention 保持原 MHA。
2. `python/sglang/srt/layers/attention/aiter_backend.py`:
   - `init_forward_metadata` 的 MLA prefill 索引更新改用 `forward_batch.seq_lens` / `forward_batch.seq_lens_sum` / `extend_seq_lens.max().item()` / `seq_lens.max().item()`(原先用 extend_prefix_lens 与各 CPU 列表),并去掉 `kv_indptr += qo_indptr` 的偏移调整;
   - `forward_extend` 给 extend 的非 MHA 情况新增显式 `else` 分支,调用 `mla_prefill_fwd`(absorb 路径,直接对 `K_Buffer` 做 MLA prefill),不再拼接 k/v 前缀后走 `flash_attn_varlen_func`;
   - `__init__` 记录 `self.enable_dp_attention = is_dp_attention_enabled()`。

关键 diff 摘录(deepseek_v2.py 派发逻辑):

```diff
-                return AttnForwardMethod.MHA
+                if is_dp_attention_enabled():
+                    if sum(forward_batch.extend_prefix_lens_cpu) == 0:
+                        return AttnForwardMethod.MHA
+                    else:
+                        return AttnForwardMethod.MLA
+                else:
+                    return AttnForwardMethod.MHA
```

作者验证:`python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000`,Accuracy: 0.937,Invalid: 0.000,Latency: 120.988 s。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响 aiter 注意力后端在 dsv3/MLA 模型上的 extend 前向:开启 DP attention 且带前缀的 extend 改走 MLA absorb 路径,其 kv_indptr/kv_indices 元数据的构造方式同步改变(`seq_lens` 语义替代 `extend_prefix_lens`);关闭 DP attention 的派发与行为保持不变。`forward_extend` 中非 absorb 分支(自定义 k/v 拼接的 MHA 路径)也改为直接使用 k_prefix/v_prefix(不再拼 extend 段),该分支其他模型形态的行为需关注。

## 关联
- https://github.com/sgl-project/sglang/issues/7692(正文给出的 issue 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
