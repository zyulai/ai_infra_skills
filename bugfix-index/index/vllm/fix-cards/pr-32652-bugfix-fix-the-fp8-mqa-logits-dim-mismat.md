# PR #32652: [Bugfix] Fix the  fp8_mqa_logits dim mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32652](https://github.com/vllm-project/vllm/pull/32652) |
| 作者 | chaunceyjiang |
| 合并日期 | 2026-01-20 |
| labels | bug, deepseek, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
fp8_mqa_logits 的 k_scale 以 [N,1] 传入但 DeepGEMM 要求 1 维 [N],flatten 后修复 DeepSeek-V3.2 报错。

## 问题现象
DeepSeek-V3.2(DSA 稀疏注意力索引器路径,`vllm serve deepseek-ai/DeepSeek-V3.2 -tp=4 --tokenizer-mode deepseek_v32`)在调用 DeepGEMM 的 fp8_mqa_logits 时发生维度不匹配(dim mismatch)报错,服务无法正常推理(issue #32647,正文未贴具体堆栈)。修复后 `python examples/online_serving/openai_chat_completion_client.py` 成功返回 chat completion(finish_reason=stop,内容正常)。

## 根因
正文披露:DeepGEMM 的 `fp8_mqa_logits` API 要求 `kv.second`(k_scales)是一维张量——正文引用其源码 `const auto& [seq_len_kv_] = get_shape<1>(kv.second);  // <----- here`,即形状必须为 `[N]`;而 vLLM 传入的 `k_scale.view(torch.float32)` 是二维 `[N, 1]`,与 API 期望的 1 维形状不符,触发 dim mismatch。

## 修复方案
- `vllm/model_executor/models/deepseek_v2.py` `sparse_attn_indexer`:把传给 fp8_mqa_logits_func 的 k_scale 拉平:
```python
logits = fp8_mqa_logits_func(
    q_fp8[chunk.token_start : chunk.token_end],
    (k_fp8, k_scale.view(torch.float32).flatten()),
    weights[chunk.token_start : chunk.token_end],
    chunk.cu_seqlen_ks,
    chunk.cu_seqlen_ke,
```
- `vllm/utils/deep_gemm.py`:同步修正 `fp8_mqa_logits` 的 docstring,`k_scales` 形状说明由 "[N] (or [N, 1])" 改为仅 "[N]"。

## 影响与回归风险
- 影响窗口:正文未说明(受影响的调用方为走 DeepGEMM / ROCM fp8_mqa_logits 的 DeepSeek-V3.2 类模型的 sparse_attn_indexer)
- 回归面:deepseek_v2.py 的 sparse attention indexer 路径(DSA);flatten 只改变形状不改数值,正常路径无风险;若其他调用方此前按 [N,1] 形状自行处理过,需确认与新的 1 维契约一致。

## 关联
- [#32647](https://github.com/vllm-project/vllm/issues/32647)(修复的 issue)
- [DeepGEMM attention.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/attention.hpp#L86)(API 依据)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
