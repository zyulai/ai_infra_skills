# PR #37501: fix: clamp dA_cumsum differences to prevent Inf in Mamba2 SSD kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37501](https://github.com/vllm-project/vllm/pull/37501) |
| 作者 | kibitzing |
| 合并日期 | 2026-03-31 |
| labels | rocm, ready |
| 热度 | 👍 6 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
Mamba2 SSD Triton kernel 对 `dA_cumsum` 差值直接 `exp()`,浮点舍入可使本应 ≤0 的差值变正而溢出为 Inf,经 SSM state 传播成 NaN;加 `tl.minimum(..., 0.0)` 钳制修复。

## 问题现象
当 Mamba2 模型的 `|A|` 值较大时,解码产生 NaN。实测(v0.15.1,RLVR 实验环境):修复前 224 个样本中 32 个出现 NaN、共 65,504 个 NaN token;修复后 NaN 样本 0/224、NaN token 0。

## 根因
PR 正文披露:大 `|A|` 下 `dA_cumsum` 会达到 float32 ULP 超出安全范围的量级;并行前缀扫描(`tl.cumsum`)引入的舍入误差可使 `(dA_cs_last - dA_cs_k)` 本应数学上恒 ≤0 的差值略微变正,`exp()` 随之溢出为 `Inf`,该 `Inf` 经 SSM state 传播,在后续 decode 步中产生 `NaN`。正文同时指出:main 上该模型/数据组合未能复现 Inf,可能是 #35397 的 kernel 重构把 `tl.exp` 换成 `fast_exp` 改变了 float32 舍入行为所致,但 main 上无保护的 `fast_exp(dA_cs_last - dA_cs_k)` 模式在数学上仍不安全,不同输入或大 `|A|` 模型可触发同样的溢出。

## 修复方案
在指数运算前对差值做非正钳制,与上游 Mamba 修复(state-spaces/mamba#713,只覆盖训练路径)对齐,应用到 vLLM 推理 kernel。受影响 kernel 两处:

`vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py` 的 `_chunk_state_fwd_kernel`:
```diff
-        scale = fast_exp(dA_cs_last - dA_cs_k) * dt_k
+        scale = fast_exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
```

`vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py` 的 `_chunk_scan_fwd_kernel`:
```diff
-        cb *= fast_exp(dA_cs_m[:, None] - dA_cs_k[None, :])
+        cb *= fast_exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(验证基于 v0.15.1;main 上因 #35397 重构未复现,但模式仍不安全)。
- 回归面:仅影响上述两个 SSD Triton kernel 的指数计算;钳制只作用于数学上应 ≤0 的差值,理论上不改变正确输出,但若上游差值本应 >0 的场景存在(正文未提及),其结果会被截断。验证数据来自 v0.15.1 环境,main 分支上的该组合未能复现问题。

## 关联
- [state-spaces/mamba#713](https://github.com/state-spaces/mamba/pull/713)(上游修复,正文引用)
- #35397(vLLM kernel 重构,正文提及,未给出链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
