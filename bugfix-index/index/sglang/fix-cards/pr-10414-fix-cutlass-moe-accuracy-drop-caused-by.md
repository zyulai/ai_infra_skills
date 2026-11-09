# PR #10414: Fix cutlass moe accuracy drop caused by attention UB from DP padding mode

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10414](https://github.com/sgl-project/sglang/pull/10414) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-09-14 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
DP attention 的 padding 模式误用 max 导致 token 被 pad,attention 拿到 padded 输入与非 padded seq lens(潜在 UB)引发 MoE 精度下降;extend batch 强制用 SUM_LEN。

## 问题现象
cutlass_moe nvfp4 精度差:换 MoE 后端(cutedsl_moe、fp8)精度正常,换 attention 后端(trtllm_mla、cutlass_mla、flashinfer)精度都不行——说明问题不在具体 kernel 而在输入准备。正文未给具体精度数字,指向 issue #10054 与 #9806 的评论。

## 根因
正文披露:直接原因是 attention core function 收到 padded 输入与非 padded 的 seq lens,属于潜在 UB(undefined behavior);根因是 DP padding mode 被设为 max 而不是 sum,max 模式会把 token pad 到各 rank 最大长度,而这些 pad token 随后进入 attention 计算。潜在影响面:任何「DP attention + 任意使用 all-gather(或 all-reduce、reduce-scatter)机制的 MoE」组合,因为这类通信机制会让 max padding 模式真正 pad token。

## 修复方案
1. `python/sglang/srt/layers/dp_attention.py` `DpPaddingMode.get_dp_padding_mode`:签名增加 `is_extend_in_batch`;当 batch 含 extend 时直接返回 `DpPaddingMode.SUM_LEN`,跳过按通信开销选择 max/sum 的原逻辑;
2. `python/sglang/srt/model_executor/forward_batch_info.py` `prepare_mlp_sync_batch`:调用处传入 `self.is_extend_in_batch`。

```diff
-    def get_dp_padding_mode(cls, global_num_tokens: List[int]) -> DpPaddingMode:
+    def get_dp_padding_mode(
+        cls, is_extend_in_batch, global_num_tokens: List[int]
+    ) -> DpPaddingMode:
+        if is_extend_in_batch:
+            return DpPaddingMode.SUM_LEN
```

## 影响与回归风险
- 影响窗口:正文未说明;触发条件为 DP attention + 相关 MoE 通信机制的 extend/混合 batch。
- 回归面:DP attention 的 padding 模式决策(`prepare_mlp_sync_batch`)——extend batch 从可能的 MAX_LEN 改为固定 SUM_LEN,decode-only batch 行为不变(仍按原代价模型选择);SUM_LEN 下的通信量与 max 模式不同(原逻辑正是为降低通信成本而选 max,extend 强制 sum 可能增加 padding 通信,正文未量化);正文自述该修复一度是「temporary workaround to check whether accuracy issue is caused by this」,最终以 extend 判定落地。

## 关联
- https://github.com/sgl-project/sglang/issues/10054#issuecomment-3289176748
- https://github.com/sgl-project/sglang/issues/9806#issuecomment-3289234555

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
