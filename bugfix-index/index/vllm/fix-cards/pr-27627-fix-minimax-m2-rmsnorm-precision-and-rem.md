# PR #27627: Fix MiniMax-M2 rmsnorm precision and remove useless code

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27627](https://github.com/vllm-project/vllm/pull/27627) |
| 作者 | rogeryoungh |
| 合并日期 | 2025-10-29 |
| labels | ready |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
rmsnorm 在 fp32 归一化后先转回原 dtype 再乘 weight 损失精度;改为 fp32 下乘完再 cast,并删除无用的 attn_window 代码。

## 问题现象
正文未描述具体现象(Test Plan 与 Test Result 留空),仅说明目的:修复 rmsnorm 精度并移除无用代码。

## 根因
正文未披露根因。(由 diff 推断)`linear_attn.py` 的 rmsnorm 实现 `x = x.to(orig_dtype) * self.weight` 在 fp32 归一化完成后先把 x 转回原 dtype(如 bf16)再乘 weight,乘法发生在低精度下,引入精度损失;改为在 fp32 下乘完 weight 再做 cast 可避免。

## 修复方案
1. `vllm/model_executor/layers/mamba/linear_attn.py` `_forward`:

```diff
-        x = x.to(orig_dtype) * self.weight
+        x = (x * self.weight).to(orig_dtype)
```

2. `vllm/model_executor/models/minimax_m2.py`:删除 `attn_window_size`/`swa_rope_theta` 的读取、list/int 类型分支与异常抛出等无用代码,不再向 `MiniMaxM2Attention` 传 `attn_window_size`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:mamba linear attention 的 rmsnorm 输出数值会与旧实现略有差异(乘法所在精度位置变化,影响所有走该实现的模型);MiniMax-M2 模型构造不再读取 `attn_window_size`/`swa_rope_theta` 配置,相关配置值被忽略。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
