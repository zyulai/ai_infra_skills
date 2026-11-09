# PR #17518: [HotFix]Fix dtype mismatch in nsa indexer on AMD device

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17518](https://github.com/sgl-project/sglang/pull/17518) |
| 作者 | Fridge003 |
| 合并日期 | 2026-01-22 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
AMD 上 NSA indexer 的 weights_proj 被硬编码为 bf16 引发 dtype 不匹配 CI 失败;改为 CUDA 用 bf16、其他设备用 fp32。

## 问题现象
AMD 设备上的 NSA(Native Sparse Attention)indexer 相关 CI 失败,失败 job:https://github.com/sgl-project/sglang/actions/runs/21192748451/job/60966055534 。正文未描述更多运行时症状(未附报错日志),按标题为 dtype mismatch。

## 根因
正文未详细披露根因分析,仅指出该 CI 失败由 PR #17205 引入,PR 标题定位为 nsa indexer 在 AMD 设备上的 dtype 不匹配。(由 diff 推断)`nsa_indexer.py` 中 `weights_proj` 的 `params_dtype` 此前被无条件设为 `torch.bfloat16`,而该模块内 `k_norm` 等计算使用 `torch.float32`;#17205 改动后在 AMD(非 CUDA)路径上 bf16 参数与 fp32 计算/内核期望的 dtype 不一致,触发失败;CUDA 路径保持 bf16 不受影响。

## 修复方案
`python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的 `__init__` 中,`weights_proj`(Linear,hidden_size→n_heads,bias=False)的 `params_dtype` 由恒为 `torch.bfloat16` 改为按设备分支:

```diff
-            params_dtype=torch.bfloat16,
+            params_dtype=torch.bfloat16 if _is_cuda else torch.float32,
```

即 CUDA 设备仍用 bf16,其他设备(含 AMD/ROCm)改用 fp32,消除 dtype 不匹配。

## 影响与回归风险
- 影响窗口:#17205 合并之后至本修复之前。
- 回归面:仅 NSA indexer 的 `weights_proj` 参数 dtype:非 CUDA 设备(AMD 等)上该投影参数从 bf16 变为 fp32,显存占用略增、精度提高;CUDA 路径 dtype 不变。NSA/DeepSeek 稀疏注意力在 AMD 上的前向计算数值会与此前不同(此前可能因 dtype 不匹配而失败或产生错误结果)。

## 关联
- CI 失败 job:https://github.com/sgl-project/sglang/actions/runs/21192748451/job/60966055534
- 引入问题的 PR: https://github.com/sgl-project/sglang/pull/17205

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
