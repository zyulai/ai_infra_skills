# PR #24511: [Multi Modal][Performance] Fused Q,K's apply_rope into one

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wwl2755 |
| 合并 | 2025-09-14 |
| 分类 | Model-specific Fast Path(major) |
| Label | ready, qwen |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 qwen2_5_vl.py 单文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/24511 |

## 一句话结论

Qwen2.5-VL 视觉注意力中 Q、K 两次 apply_rope 合并为一次 concat 调用(rope 耗时 731→378 µs),TTFT 整体改善约 11%。

## 动机

Qwen2.5-VL 的 vision encoder 中,Q 和 K 各自独立调用一次 `apply_rotary_pos_emb_vision`,两次独立的 kernel 调用重复处理同一份 rotary 位置编码;此外 flash_attn 路径上还串联了多次 `rearrange` 形变操作,构成视觉 prefill 的额外固定开销。视觉 token 数量大时,这些小 kernel 的累计时间直接抬升 TTFT。

## 优化点

将 Q、K 沿第 0 维 concat 成一个 `[2*b, s, heads, head_dim]` 张量,用一次 `apply_rotary_pos_emb_vision` 完成旋转位置编码,再 `torch.chunk` 拆回 q、k,把两次 rope 调用合并为一次。同时合并 flash_attn 路径的形变链:原先 `rearrange(output, "(b s) ... -> b s ...")` 与末尾 `rearrange(context_layer, "b s h d -> s b (h d)")` 两步,直接合并为一步 `"(b s) h d -> s b (h d)"` 并 contiguous,减少中间拷贝。SDPA 与 xformers 分支保持各自逻辑不变。

## 关键改动

```text
vllm/model_executor/models/qwen2_5_vl.py: vision attention forward 中合并 Q/K rope 调用与 flash_attn 后处理 rearrange
```

关键片段:

```python
qk_concat = torch.cat([q, k], dim=0)
qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
q, k = torch.chunk(qk_rotated, 2, dim=0)
```

## 性能声明

- Kernel 级:Qwen2.5-VL 的 Q/K apply_rope 时间 731 µs → 378 µs,GPU 时间 485 µs → 255 µs;flash_attn 路径合并 rearrange 再省约 12 µs。PR 未注明上述微基准的硬件与输入规模。
- E2E:`vllm serve Qwen/Qwen2.5-VL-7B-Instruct`,`vllm bench serve` openai-chat、VisionArena-Chat 数据集(train split)、num-prompts 100、hf-output-len 1、seed 40。Mean TTFT 6493.82 → 5857.08 ms,Median TTFT 6356.22 → 4647.33 ms,Request throughput 7.80 → 8.01 req/s;PR 自述整体约 11% TTFT 改善。PR 未注明测试硬件。
- Correctness:lm_eval mmmu_val(5-shot、limit 250)两侧 acc 均为 0.4611。

## 验证与回归风险

依赖 mmmu_val 精度对照与 serve benchmark,diff 中未新增单元测试。风险面:仅触及 Qwen2.5-VL 视觉路径;`torch.cat` 引入一次额外显存拷贝,以换取更少的 kernel launch,属于时间换空间权衡;改动位于共享的 vision attention forward,影响 flash_attn、SDPA、xformers 三种 backend 分支的输出排布路径(flash_attn 的 rearrange 合并改变了 context_layer 的生成方式,需保证 contiguous)。作者表示如可行会推广到其他模型。

## 关联

相关 issue:#23880、#23884(正文链接)。同区域(多模态视觉编码性能)后续工作:#61458(批量 PaddleOCR-VL 视觉 encoder)。无同文件前序 PR 记录。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
