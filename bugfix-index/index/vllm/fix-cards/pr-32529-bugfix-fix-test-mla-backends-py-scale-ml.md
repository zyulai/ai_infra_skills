# PR #32529: [BUGFIX] Fix `test_mla_backends.py`. Scale MLA projection weights to prevent numerical instability

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32529](https://github.com/vllm-project/vllm/pull/32529) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-01-19 |
| labels | bug, ready, v1 |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
MLA 后端对比测试的随机权重使投影输出 std≈√512≈22.6,LSE 合并数值失稳导致误报;按 1/√kv_lora_rank 缩放权重根治(#32484 为临时绕过)。

## 问题现象
V1 Test attention(B200)CI run 失败。正文定性:失败源于数值精度损失导致的虚假失败(spurious failures),并非后端真实 bug。缩放后测试在真实量级输入下运行,各 MLA 后端可被公平比较。

## 根因
正文明确披露:测试使用随机权重矩阵,`kv_lora_rank=512` 的输入使投影输出标准差约 √512 ≈ 22.6;如此大的幅值造成极端 attention 分数,破坏跨 chunk 的 LSE(log-sum-exp)合并稳定性,进而引发测试失败。#32484 已作为 workaround 合入,本 PR 是根因修复:对 `kv_b_proj_weight` 施加 `1/sqrt(kv_lora_rank)` 缩放,使投影输出具有单位方差量级。

## 修复方案
`tests/v1/attention/test_mla_backends.py` 的 `test_backend_correctness`:在随机生成 `W_UK`、`W_UV` 之后、拼接为 `kv_b_proj_weight` 之前进行缩放:

```diff
+    # Scale weights to produce realistic magnitude outputs.
+    # Without scaling, projection output has std ~sqrt(kv_lora_rank) ≈ 22.6,
+    # causing extreme attention scores and numerical instability in LSE merging.
+    weight_scale = 1.0 / (kv_lora_rank**0.5)
+    W_UK = W_UK * weight_scale
+    W_UV = W_UV * weight_scale
+
     kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)
```

## 影响与回归风险
- 回归面:仅测试文件 `tests/v1/attention/test_mla_backends.py`;测试输入分布改变后,各 MLA 后端的比较在更接近真实的数值范围内进行,原 workaround(#32484)可被根因修复取代;产品代码无任何改动。

## 关联
- https://github.com/vllm-project/vllm/pull/32484 (正文提到的已合入 workaround,本 PR 为根因修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
