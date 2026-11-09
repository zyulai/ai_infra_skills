# PR #6734: Fix incorrect LoRA weight loading for fused gate_up_proj

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6734](https://github.com/sgl-project/sglang/pull/6734) |
| 作者 | lifuhuang |
| 合并日期 | 2025-05-31 |
| labels | (none) |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
phi-4-mm 支持 PR 引入两处错误:fused LoRA gate_up_proj 形状靠广播掩盖、Idefics seqlens 计算错误;修复后 MMMU 47→53.1。

## 问题现象
作者在测试中发现此前支持 phi-4-mm 的 PR 引入两个 bug:主 bug 是 fused LoRA 权重未正确处理 gate_up_proj 形状,此前未被发现是因为 PyTorch 通过广播(broadcast)隐式容忍了形状不匹配;次 bug 是对 Idefics embedding 的 seqlens 计算理解有误——当 `tgt_sizes` 缺失时,input_embeds 不应被 mask。影响:phi-4-mm 多模态模型加载 LoRA 后精度明显偏低,MMMU 仅 47 分。修复后 SGLang MMMU 升至 53.1,与 vLLM(52.5)在随机误差内一致,略低于原论文报告的 55(作者认为可能源于采样参数或评测脚本差异)。

## 根因
正文披露:
1. (主要)前一个 PR 引入 fused LoRA 权重支持时,没有正确处理 gate_up_proj 的形状;`normalize_gate_up_proj` 中只对 `lora_A` 做 `repeat(2, 1)` 复制,却假定 `lora_B`「已堆叠、无需操作」,实际 lora_B 未按 gate/up 两半堆叠成三维,形状不匹配被 PyTorch 广播掩盖。
2. (次要)对 Idefics embedding seqlens 计算理解错误:`tgt_sizes` 缺失时错误地用 patch_attention_mask 计算 patch_len,而按 vLLM 参考(idefics2_vision_model.py)此时应直接使用 input_embeds 的长度,不应引入 mask。

## 修复方案
1. `python/sglang/srt/lora/lora.py` `normalize_gate_up_proj`:对 `lora_B` 权重按 output_dim 上下两半沿 dim=0 `torch.stack`,使其符合当前 fused gate_up_proj 约定。
2. `python/sglang/srt/models/idefics2.py` `compute_cu_seqlens`:重写——有 `tgt_sizes` 时用 `tgt_sizes[:,0]*tgt_sizes[:,1]`,否则用 `input_embeds` 的 (batch, seq) 全填充;两者都缺则 raise;forward 传入 `hidden_states` 而非 `patch_attention_mask`。
3. diff 还包含:`python/sglang/srt/conversation.py` phi-4-mm 模板的 system_message 清空、system_template 改为 `{system_message}`;`python/sglang/srt/models/phi4mm.py` `should_apply_lora` 返回值改为 `bool(...)`(正文 Modification 未单独提及这两处,属 diff 附带改动)。

```diff
-                # else: "lora_B" is already stacked, no operations is needed.
+                else:
+                    output_dim = weights[gate_up_name].shape[0] // 2
+                    weights[gate_up_name] = torch.stack(
+                        [
+                            weights[gate_up_name][:output_dim, :],
+                            weights[gate_up_name][output_dim:, :],
+                        ],
+                        dim=0,
+                    )
```

## 影响与回归风险
- 影响窗口:引入 phi-4-mm 支持的前一 PR 合并之后至本修复之前。
- 回归面:fused gate_up_proj 的 LoRA 权重加载路径(lora_B 增加一次 stack,改变张量布局,影响使用 fused LoRA 的模型);Idefics2/phi-4-mm 视觉编码器 cu_seqlens 计算;phi-4-mm 会话模板(prompt 格式变化会改变生成行为);phi4mm LoRA 模块匹配的返回类型。

## 关联
- vLLM 参考实现:https://github.com/vllm-project/vllm/blob/fd7bb88d72ba721d6eb4f9d34198ad930c36c177/vllm/model_executor/models/idefics2_vision_model.py#L34
- 其余 issue/PR:正文未列出(未给出前一 PR 编号)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
