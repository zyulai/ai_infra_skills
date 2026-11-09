# PR #47079: [Bugfix][MLA] Fix LSE log-base mismatch in DCP + FlashInfer MLA decode

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47079](https://github.com/vllm-project/vllm/pull/47079) |
| 作者 | GirasoleY |
| 合并日期 | 2026-06-30 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
FlashInfer trtllm-gen MLA 返回 log2 基 LSE,DCP combine 硬编码自然对数致 softmax 分母错误;为 backend 增加 lse_base_on_e 能力标志并透传 combine。

## 问题现象
Kimi-K2.5-NVFP4 + FP8 KV + DCP=4 + `FLASHINFER_MLA` 下:同一 prompt 在一次运行内发送两次,24/32 个位置的 argmax token 不同(同 backend、无 DCP 基线的噪声底约 4/32);所有测试 prompt 的 argmax 都偏离无 DCP 基线。`TRITON_MLA`、FlashAttention MLA、FlashMLA、Cutlass MLA 不受影响——它们的 kernel 输出自然对数 LSE,恰与旧的硬编码假设一致。修复后 5-prompt greedy refcomp 0 token mismatch(修复前 28+24+24+4+0),运行内稳定性 24/32 → 5/32(与基线持平);GSM8K 5-shot 相对 TEP4 基线 delta +0.07%/+0.08%。

## 根因
正文明确披露:FlashInfer 的 trtllm-gen MLA decode kernel 返回的 LSE 是 log2 基(见 flashinfer 参考实现 `flashinfer/trace/templates/attention.py:81` 的 `logsumexp / log(2.0)`),但 `MLAAttention` 的 DCP combine 在调用 `cp_lse_ag_out_rs` / `dcp_a2a_lse_reduce` 时硬编码 `is_lse_base_on_e=True`,断言自然对数。底数不匹配使跨 shard 合并时的逐 shard 重加权用错底,产生错误的 softmax 分母与损坏的 attention 值。这是 #43729("Support DCP with FlashInfer MLA")合入后的遗留 bug:该 PR 打通了 LSE 返回使 DCP combine 得以运行,但没做 log 基对齐。

## 修复方案
- `vllm/v1/attention/backend.py`:`AttentionImplBase` 新增能力标志 `lse_base_on_e: bool = True`(True=自然对数,False=log2),注释说明 DCP combine kernel 经 `IS_BASE_E` constexpr 依赖该值;
- `vllm/v1/attention/backends/mla/flashinfer_mla.py`:`FlashInferMLAImpl` 覆写 `lse_base_on_e: bool = False`;
- `vllm/model_executor/layers/attention/mla_attention.py`:DCP combine 两处调用由 `is_lse_base_on_e=True` 改为 `is_lse_base_on_e=self.impl.lse_base_on_e`。

```diff
-                        is_lse_base_on_e=True,
+                        is_lse_base_on_e=self.impl.lse_base_on_e,
```

kernel 侧 `IS_BASE_E` 原生支持两种底(exp/exp2),无 FP 乘法或舍入损失;未来新增 log2 backend 只需一行 ClassVar。

## 影响与回归风险
- 回归面:DCP(decode context parallelism)+ FlashInfer MLA 的 LSE 合并路径;其他 MLA backend 使用默认 True,行为不变;MLAAttention 依赖 impl 提供该属性,自定义 AttentionImpl 子类需注意新标志的默认值。

## 关联
- https://github.com/vllm-project/vllm/pull/43729 (遗留 bug 来源)
- https://github.com/vllm-project/vllm/pull/47074 (同一文件的正交修复,非重复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
