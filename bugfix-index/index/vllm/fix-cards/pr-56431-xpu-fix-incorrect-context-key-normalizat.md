# PR #56431: [XPU] Fix incorrect context-key normalization for Qwen DFlash-based models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56431](https://github.com/vllm-project/vllm/pull/56431) |
| 作者 | yma11 |
| 合并日期 | 2026-09-17 |
| labels | intel-gpu, qwen, dflash |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
XPU rms_norm kernel 把 2D 逐层 K-norm 权重当 1D 用,除第 0 层外草稿层 K 值归一化错误;在 qwen3_dflash.py 增加 XPU 逐层 fallback 修复。

## 问题现象
XPU 平台上基于 DFlash 的 Qwen 模型(如 DSpark 用 `Qwen/Qwen3-4B-FP8` + draft `deepseek-ai/dspark_qwen3_4b_block7`)精度与接受率严重异常:GSM8K 准确率仅 0.487(参考 0.801,门槛 0.801*0.95,断言失败),acceptance_rate 仅 0.011,acceptance_len 1.080(参考 0.428 / 3.994)。修复后:gsm8k_accuracy=0.859、acceptance_rate=0.459、acceptance_len=4.215(1319 题,3:51 完成,输出 661.79 toks/s)。另附 `Qwen/Qwen3.8-27B-FP8` + `z-lab/Qwen3.8-27B-DFlash2`(TP=4,dflash,3 个投机 token)的 bench serve 前后对比截图。

## 根因
PR 正文披露:DFlash 把逐层 K-norm 权重堆叠并调用 `rms_norm`,输入形状为 `[num_layers, num_context_tokens, num_kv_heads, head_dim]`,权重形状为 `[num_layers, head_dim]`。vLLM 当前使用的 XPU kernel 把这个 2D 权重当作 1D 向量处理,始终应用第一层的权重行;因此除第一层外的所有草稿层(draft layers)的 context K 值在写入 KV cache 之前都用错误的权重做了归一化。

## 修复方案
在 `vllm/model_executor/models/qwen3_dflash.py` 的 `_normalize_context_k` 中加入 XPU 专用的逐层 fallback:检测到 `current_platform.is_xpu()` 时按层循环,每层用对应的 `self._k_norm_weights[layer_idx]` 调用 `ops.rms_norm`,保证逐层权重被正确应用(临时方案;正确的 kernel 将随下一次 kernel 发布提供,见 vllm-xpu-kernels PR 579):

```diff
+        if current_platform.is_xpu():
+            for layer_idx in range(all_k.shape[0]):
+                ops.rms_norm(
+                    all_k_normed[layer_idx],
+                    all_k[layer_idx],
+                    self._k_norm_weights[layer_idx],
+                    self._rms_norm_eps,
+                )
+            return all_k_normed
```

非 XPU 平台保持原批量 `ops.rms_norm` 调用不变。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是 XPU 平台上使用 DFlash 逐层 K-norm 的 Qwen 模型(Qwen3 DFlash / DSpark 类配置)。
- 回归面:改动仅限 `qwen3_dflash.py` 的 context K 归一化路径,且只在 XPU 平台分支生效;XPU 上逐层 Python 循环调用 `ops.rms_norm` 可能带来一定性能开销(正文未量化该 fallback 的性能损失);非 XPU 平台不受影响。验证:`test_dspark.py::test_dspark_correctness_and_acceptance_rate[qwen3-deepspec]` 由失败转为通过。

## 关联
- https://github.com/vllm-project/vllm-xpu-kernels/pull/579(正式 kernel 修复将随下一次 kernel 发布)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
