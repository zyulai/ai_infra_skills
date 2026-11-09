# PR #23864: [BugFix][AMD][Deepseek] fix a dtype mismatch error for deepseek running on AMD

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23864](https://github.com/vllm-project/vllm/pull/23864) |
| 作者 | KingsleyZhang123 |
| 合并日期 | 2025-08-29 |
| labels | rocm, ready, deepseek |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
AMD grouped_topk 缺 dtype 转换,DeepSeek MoE 的 float32 correction_bias 撞上 bf16 gating_output 启动即报错;传入前按 gating_output 转换。

## 问题现象
DeepSeek V3 在 AMD 机器上 worker 启动阶段即失败,报 `RuntimeError: gating_output.dtype() == correction_bias.dtype()`;修复后 worker 创建成功。修复后性能:QPS 0.67、平均 TTFT(客户端)673.22ms;GSM8K 8-shot 评测 em/f1 = 0.980000。

## 根因
正文披露:commit fdeb3da 把 `e_score_correction_bias` 的 dtype 改为 float32。这在 H100 上没问题,因为执行 `grouped_topk` 前已做过 dtype 转换(fused_moe.py);但 AMD 实现的 grouped_topk 没有这一步转换,而 `gating_output` 默认是 bfloat16,于是触发类型检查错误。

## 修复方案
`vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` 的 `rocm_aiter_grouped_topk`:调用 `torch.ops.vllm.rocm_aiter_biased_grouped_topk` 前,把 correction_bias 转成与 gating_output 一致的 dtype(另有 docstring 尾随空格清理):

```diff
-            e_score_correction_bias,
+            e_score_correction_bias.to(gating_output.dtype),
```

## 影响与回归风险
- 影响窗口:自 commit fdeb3da 将 `e_score_correction_bias` 转 float32 起,AMD 的 biased grouped_topk 路径受影响(正文说明)。
- 回归面:仅 `rocm_aiter_fused_moe.py` 的 `rocm_aiter_grouped_topk` biased 分支:correction_bias 改为按 gating_output 的 dtype(bf16 场景即 bf16)传入 AITER 算子;docstring 空白清理无行为影响。

## 关联
- commit [fdeb3da](https://github.com/vllm-project/vllm/commit/fdeb3dac132c9ef92d981dd811529e6496781b07)(引入 dtype 变更)
- fused_moe.py 中 H100 路径的 dtype 转换(正文引用的代码位置)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
