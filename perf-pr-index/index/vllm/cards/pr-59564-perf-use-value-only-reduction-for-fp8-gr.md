# PR #59564: [Perf] Use value-only reduction for FP8 group quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @porridgewithraisins |
| 合并 | 2026-11-16 |
| 分类 | Quantization（tier: major） |
| Label | ready, torch.compile, quantization, verified |
| Diff | pack 标注 +None/-None, None files（实际 diff 仅 `input_quant_fp8.py` 一行替换）；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/59564 |

## 一句话结论

把 torch.compile 路径的 FP8 分组量化中 `torch.max(dim)[0]` 改为 `amax()`，消除 Inductor 生成 kernel 里多余的 value+index 双归约，编译路径从比 CUDA 慢约 10% 变为最快快 20%。

## 动机

Issue #25094：`QuantFP8` 的 compiled（torch.compile / native）实现与 CUDA 实现结果一致，但性能更差。原因很直接：编译版用 `torch.max(...)[0]`，Inductor 会为它生成同时对 values 和 indices 归约的 kernel（`triton_helpers.max_with_index`），而 CUDA 实现只归约 values。早期怀疑是 autotune 问题（小 num_tokens 如 1024 时 inductor 选择了更慢的 XBLOCK=32 而非 XBLOCK=8），但 8192/16384 等 token 数下确实用了 XBLOCK=8，问题出在生成 kernel 内不必要的 index 归约。

## 优化点

- 一行改动：`absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()` 改为 `absmax = x_grouped.abs().amax(dim=-1, keepdim=True).float()`。
- Inductor 生成的 kernel 由 `triton_helpers.max_with_index(tmp4, tmp5, 1)`（广播 index 参与归约）变为 `triton_helpers.max2(tmp4, 1)`：静态指令 480 → 360、SHFL.BFLY 6 → 3、Integer SEL 33 → 17（寄存器 29 → 30 基本不变）。PR 判断慢不是寄存器压力，而是多余指令所致。
- bench repro 用 `num_tokens` 最大值编译（`torch.compile(fullgraph=True, dynamic=False)` + `mark_dynamic`），即按 slack 讨论以 compile max 编译；同时验证改动输出与前一实现完全一致（`assert_close(..., rtol=0, atol=0)`，输出与 scales 均为 0 容差）。

## 关键改动

```python
# vllm/model_executor/layers/quantization/input_quant_fp8.py, _quantize_group_native
-        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
+        absmax = x_grouped.abs().amax(dim=-1, keepdim=True).float()
```

生成代码对比（PR 正文摘录）：

```py
# max()[0] -> Inductor generated
  tmp5 = tl.broadcast_to(rindex, tmp4.shape)
  tmp3_val, tmp3_idx = triton_helpers.max_with_index(tmp4, tmp5, 1)
  tmp3 = tmp3_val[:, None]
# amax() -> Inductor generated
  tmp5 = triton_helpers.max2(tmp4, 1)[:, None].to(tl.float32)
```

## 性能声明

（环境：H200 sm90，PyTorch 2.13.0+cu130，Triton 3.7.1，driver 580.126.09，vLLM 2fc3f9；bench 用 `triton.testing.do_bench_cudagraph` 1000 次放大，QuantFP8(False, GroupShape(1, 128), column_major_scales=False)，use_ue8m0 断言为真。）

- 总结（PR 原文）：Before 约 10% slower than CUDA；After 最快 up to 20% faster than CUDA。
- compile max 8192 组：512×2048：2.766 → 2.497 µs（CUDA 2.603）；8192×2048：24.500 → 17.734（22.854）；512×4096：4.436 → 3.541（3.971）；8192×4096：48.129 → 34.885（43.664）；512×7168：6.513 → 4.918（5.871）；8192×7168：82.865 → 59.053（74.613）。
- compile max 16384 组：512×2048：2.954 → 2.514（2.601）；16384×2048：48.107 → 34.819（43.671）；512×4096：4.480 → 3.540（3.973）；16384×4096：94.161 → 66.879（85.002）；512×7168：6.416 → 4.903（5.890）；16384×7168：162.552 → 115.090（146.931）。

## 验证与回归风险

bench repro 在每组形状上先做 `torch.testing.assert_close(new[0].float(), old[0].float(), rtol=0, atol=0)` 与 scales 的 0 容差对比（改动与原实现输出逐位一致）。回归风险：仅触及 `_quantize_group_native` 一个函数的 absmax 归约；`max(dim)[0]` 与 `amax()` 在该语义下等价（同 dim、keepdim，仅丢弃 index），CUDA 路径不变；padding 分支（F.pad 后分组）沿用同一表达式。PR checklist 中 Design Fit / Testing / Code Quality / Contents 四项未勾选，`/pr-checklist` 未使用；正文未列 CI 运行结果。pack 未提供端到端模型级验证数据。

## 关联

- Fixes #25094。
- Slack 讨论：https://vllm-dev.slack.com/archives/C08K1FAHFPH/p1790784879162549 。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
