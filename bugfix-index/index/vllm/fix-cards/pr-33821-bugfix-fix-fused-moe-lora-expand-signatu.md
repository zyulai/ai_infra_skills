# PR #33821: [Bugfix] Fix _fused_moe_lora_expand signature mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33821](https://github.com/vllm-project/vllm/pull/33821) |
| 作者 | xyang16 |
| 合并日期 | 2026-02-07 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

_fused_moe_lora_expand 删参后 fake 函数未同步,签名不匹配,同步移除 b_intermediate_cache1。

## 问题现象

正文未描述用户可感知的运行期症状或报错日志;问题本身是自定义算子 `_fused_moe_lora_expand` 与其对应的 fake 函数 `_fused_moe_lora_expand_fake` 之间的函数签名不匹配,属于 LoRA fused MoE Triton 算子注册层面的一致性问题。

## 根因

按正文披露:`_fused_moe_lora_expand` 中已移除参数 `b_intermediate_cache1`,但 fake 函数 `_fused_moe_lora_expand_fake` 没有同步删除该参数,导致两个函数签名不匹配。(由 diff 推断)fake 函数是真实算子的形状推断/注册对等物,二者参数表必须逐一对齐;真实算子删参后 fake 侧多出的 `b_intermediate_cache1: torch.Tensor` 使调用注册路径拿到的签名与实现不一致。

## 修复方案

修改 `vllm/lora/ops/triton_ops/fused_moe_lora_op.py`:从 `_fused_moe_lora_expand_fake` 的参数列表中删除 `b_intermediate_cache1: torch.Tensor` 一行,使其与 `_fused_moe_lora_expand` 的签名恢复一致。其余参数(output、a_intermediate_cache1、lora_b_stacked、topk_weights、sorted_token_ids 等)保持不变。

关键 diff 摘录:

```diff
 def _fused_moe_lora_expand_fake(
     output: torch.Tensor,
     a_intermediate_cache1: torch.Tensor,
-    b_intermediate_cache1: torch.Tensor,
     lora_b_stacked: list[torch.Tensor],
     topk_weights: torch.Tensor,
     sorted_token_ids: torch.Tensor | None,
```

## 影响与回归风险

- 影响窗口:自真实算子移除 `b_intermediate_cache1` 的那次改动起至本修复前;正文未给出具体版本,不写。
- 回归面:仅 fake 函数签名一处,删除的是真实算子已不存在的参数,真实算子实现未动;使用 fused MoE LoRA 的路径在算子注册/形状推断上恢复一致,无行为收窄。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
