# PR #22931: [Fix/Kernel] Add JIT rmsnorm_hf kernel to fix transformers backend MMLU accuracy regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22931](https://github.com/sgl-project/sglang/pull/22931) |
| 作者 | Jiminator |
| 合并日期 | 2026-04-23 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 3 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
transformers 后端用 sgl_kernel.rmsnorm 替换 HF RMSNorm,舍入顺序不同,int4wo-128 下 MMLU 从 0.703 掉到 0.656;新增 HF 语义的 JIT rmsnorm_hf 恢复精度。

## 问题现象
transformers 后端模型 MMLU 精度回归:commit `34ddf135fd` 引入 `recursive_replace()` 后,在 int4wo-128 量化下 32 层 transformer 的 ~1-ULP 舍入差复合,翻转 3 个边界 MMLU 题,H100 上 0.703 → 0.656(Before Fix: 0.6562)。#22435 的朴素修复(直接回退)带来明显吞吐回归(868.824 → 780.230 tok/s)。

## 根因
正文披露:`sgl_kernel.rmsnorm` 与 HF `LlamaRMSNorm` 的舍入顺序不同——前者在 fp32 中乘 weight 再 cast,而 HF 先把归一化输出 cast 到激活 dtype 再乘 weight;int4wo-128 量化下这个 ~1-ULP 差异跨 32 层复合,足以翻转边界样本。

## 修复方案
- 新增 JIT kernel `rmsnorm_hf`(csrc/elementwise/rmsnorm_hf.cuh + python wrapper):512 线程标量跨步 CTA 与 warp 两种 launch config,精确复刻 HF「先 cast 后乘 weight」顺序;寄存器缓存输入以减半全局读,支持 fp16/bf16、PDL,hidden_size 为 512 倍数(覆盖 Llama/Qwen/Mistral)。
- `layernorm.py`:`cast_x_before_out_mul=True` 且 residual 为 None 时派发到该内核,不支持时回退 `forward_native`;batch-invariant 模式下该类层也走 native。
- `models/transformers.py`:`replace_rms_norm_class` 非 Gemma 路径设置 `cast_x_before_out_mul=True`。

```python
+      const Float xn = cast<Float>(xi_cache[k] * rstd);
+      yr[i] = cast<Float>(static_cast<float>(xn) * static_cast<float>(wr[i]));
```

## 影响与回归风险
- 影响窗口:正文未说明(commit 34ddf135fd 引入回归,#22435 为首次修复尝试)
- 回归面:transformers 后端非 Gemma 模型的 RMSNorm 前向;新增内核仅在 CUDA、支持 hidden size、fp16/bf16 且 weight 同 dtype 时启用,否则走原 native 路径。正文实测修复后吞吐 845.62 tok/s(优于朴素回退的 780.230)。

## 关联
- #22435(先前的朴素修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
