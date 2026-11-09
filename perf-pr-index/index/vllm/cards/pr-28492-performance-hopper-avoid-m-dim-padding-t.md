# PR #28492: [Performance][Hopper] Avoid M dim padding to 4x for most cases (due to cuda graphs paddings)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2025-11-12 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | ready, deepseek, nvidia |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际仅改动 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/28492 |

## 一句话结论

Hopper CUTLASS FP8 GEMM 在 M 维无需补齐时跳过 padding,DSR1-FP8 8xH200 TPOT 降低 3.3%。

## 动机

Hopper 上的 CUTLASS FP8 blockwise GEMM 要求 M 维(num_tokens)补齐到 4 的倍数。此前 `_padded_cutlass` 无论是否需要都会执行 padding 路径:分配 zero 张量、拷贝输入、跑 GEMM、再切回原形状。而在使用 CUDA Graph 的场景下,capture sizes 本身已经把 num_tokens pad 到 4x,大部分调用实际已满足对齐要求,这层无谓的 padding 拷贝成为纯开销。

## 优化点

改动在 `_padded_cutlass` 内部:先计算 padded 尺寸并判断 `has_pad = padded > dim`。只有当输入确实需要补齐时才走原来的分配-拷贝-padded GEMM-裁剪路径;否则直接把原始 `qx`/`x_scale` 传给 `cutlass_scaled_mm`,省去 zero 初始化、两次 copy_ 和输出切片。由于 CUDA Graph capture sizes 通常已保证 M 是 4 的倍数,"most cases" 下会命中免 padding 的直达路径。机制上等价于把"无条件 pad"改成"按需 pad",数值语义不变。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/fp8_utils.py: _padded_cutlass 增加 has_pad 分支,免 padding 时直接调用 cutlass_scaled_mm
```

关键代码摘录(节选,<=15 行):

```python
has_pad = padded > dim
if has_pad:
    padded_qx = torch.zeros(padded_shape, ...)
    padded_qx[0 : qx.shape[0], ...].copy_(qx)
    ...
    output = cutlass_scaled_mm(padded_qx, weight, padded_x_scale, ...)
    return output[0 : qx.shape[0], ...]
else:
    return cutlass_scaled_mm(qx, weight, x_scale, weight_scale, block_size, output_dtype)
```

## 性能声明

- DeepSeek-R1-FP8,8xH200,batch size 32:Mean TPOT 31.41 ms → 30.40 ms,即 3.3% TPOT 改善。
- pack 素材未提供更详细的 workload(如输入长度分布)信息,TPOT 条件以 PR 正文所述为准。

## 验证与回归风险

- `_padded_cutlass_fake` 对应的 fake 实现路径保持存在,torch.compile 下的 shape 推导不受影响;diff 中未包含新增测试。
- 影响路径:所有走 CUTLASS blockwise FP8 GEMM 且启用该 padding helper 的模型(如 DeepSeek 系列)在 Hopper 上的 dense/attention GEMM 调用。若 M 非 4 倍数仍会 pad,正确性风险集中在 `has_pad` 判断与对齐要求的一致性上;非对齐输入仍走旧路径,回归风险较低。

## 关联

- 同为 CUDA Graph capture padding 与 GEMM shape 交互的优化;PR 未在正文中列出前序/后续 PR,标签含 deepseek、nvidia。
- 关联 PR 未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
