# PR #26063: [BugFix] Fix FI accuracy issue when used for MLA prefill

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26063](https://github.com/vllm-project/vllm/pull/26063) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-10-02 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
FlashInfer 做 MLA prefill 时返回的 LSE 布局与下游预期相反导致精度崩溃,补上 transpose 修复。

## 问题现象
以 `VLLM_LOGGING_LEVEL=DEBUG vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` 启动(日志显示 `Using FlashInfer prefill for MLA`),运行 GSM8K 评测(1319 题、5-shot):main 分支 Accuracy 仅 0.208、Invalid responses 0.006;应用本修复后 Accuracy 恢复到 0.782、Invalid responses 0.000。即 FlashInfer(FA3)用于 MLA prefill 时模型输出大面积错误。吞吐基本不变(49.12 vs 46.48 qps),说明是纯精度问题。

## 根因
正文未披露根因,仅以精度对比数据呈现问题。(由 diff 推断)FlashInfer 的 prefill 接口在 `return_lse=True` 时返回的 softmax_lse 布局为 `(q_len, num_heads)`,而 MLA common 层的下游逻辑(如按 chunk 合并 prefill 结果)期望 `(num_heads, q_len)`;两处 FlashInfer 调用点未做转置,沿用了错误布局的 LSE,导致注意力输出合并出错、精度崩塌。

## 修复方案
修改 `vllm/v1/attention/backends/mla/common.py` 中两个 FlashInfer prefill 调用点,将返回的 LSE 从 `(q_len, num_heads)` 转置为 `(num_heads, q_len)` 并 `.contiguous()`:
- `_run_prefill_new_tokens_fi`:先接住返回值,`isinstance(ret, tuple)` 时对 `ret[1]` 做 `transpose(0, 1).contiguous()` 后再返回;
- `_run_prefill_context_chunk_fi`:改为 `attn_out, lse = prefill.prefill_chunks[chunk_idx].run(..., return_lse=True)`,返回 `attn_out, lse.transpose(0, 1).contiguous()`。

```diff
+        if isinstance(ret, tuple):
+            # Convert from (q_len, num_heads) to (num_heads, q_len)
+            return ret[0], ret[1].transpose(0, 1).contiguous()
+        return ret
```

CUDNN/FA 等其他后端的对应函数未改动。

## 影响与回归风险
- 影响窗口:正文未给出引入问题的提交或版本范围,不予推断。
- 回归面:仅影响走 FlashInfer MLA prefill 的路径(main prefill 与 context chunk 两个函数);该转置对返回值布局敏感,若 FlashInfer 后续版本改变 LSE 布局约定,此处需同步调整。CUDNN、FA chunk 路径不受影响。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
