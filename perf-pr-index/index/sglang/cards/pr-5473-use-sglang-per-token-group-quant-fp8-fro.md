# PR #5473: use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @strgrb |
| 合并 | 2025-04-18 |
| 分类 | Quantization (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/5473 |

## 一句话结论

DeepGEMM 的 W8A8 block-FP8 量化入口从 Triton `per_token_group_quant_fp8` 换成 sgl-kernel 的 CUDA `sglang_per_token_group_quant_fp8`,DeepSeek-R1 在 8xH20 上吞吐约 +5%。

## 动机

`apply_w8a8_block_fp8_linear` 在启用 JIT DeepGEMM 时,激活量化走的是 Triton 实现的 `per_token_group_quant_fp8`。sgl-kernel 已提供更快的 CUDA 实现 `sgl_per_token_group_quant_fp8`(由 #4817 引入),本 PR 把 DeepGEMM 路径切换过去。作者用 2048x7168 的 fp8 量化对 kernel 做了对比测试(附图),并在 DeepSeek-R1 上看到约 5% 的 e2e 吞吐提升。

## 优化点

两处机制改动:(1) `fp8_utils.py` 的 DeepGEMM 分支把 `per_token_group_quant_fp8(...)` 换成 `sglang_per_token_group_quant_fp8(...)`,其余参数(`block_size[1]`、`column_major_scales=True` 等)不变——即用 sgl-kernel 的 CUDA kernel 完成 per-token-group 的 FP8 量化与 scale 计算,替代 Triton kernel。(2) `fp8_kernel.py` 的 `sglang_per_token_group_quant_fp8` wrapper 增加 `column_major_scales` 与 `scale_tma_aligned` 两个参数:column-major 时按 `(group_num, ...) + permute` 构造列主序 scale 张量;`scale_tma_aligned` 时再对 token 维对齐到 4 x sizeof(float)(aligned_size 向上取 4),`permute(-1, -2)` 后切片——这是 CUDA kernel 输入所需的列主序 scale 布局,与 DeepGEMM 的 TMA 对齐要求匹配。非 column-major 分支保持原有行主序布局。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_utils.py: DeepGEMM 分支的激活量化改调 sglang_per_token_group_quant_fp8
python/sglang/srt/layers/quantization/fp8_kernel.py: wrapper 新增 column_major_scales/scale_tma_aligned 参数并构造列主序(可选 TMA 对齐)scale
```

关键片段(列主序 + TMA 对齐的 scale 构造):

```python
if column_major_scales:
    if scale_tma_aligned:
        aligned_size = (x.shape[-2] + 3) // 4 * 4
        x_s = torch.empty(
            x.shape[:-2] + (x.shape[-1] // group_size, aligned_size), ...
        ).permute(-1, -2)[: x.shape[-2], :]
```

## 性能声明

- kernel 级:2048x7168 fp8 量化的新旧 kernel 对比(PR 附截图,数字在图内,正文未以文本给出)
- e2e:DeepSeek-R1 在 8xH20 上,约 5% 吞吐提升
- PR 未注明更详细的 benchmark 条件(请求负载、并发等)

## 验证与回归风险

PR 未注明新增单元测试与准确性评估;Checklist 各项均未勾选。回归面:仅 `_enable_jit_deepgemm` 的 DeepGEMM W8A8 block-FP8 路径改走 CUDA kernel;Triton `per_token_group_quant_fp8` 仍存在、其他调用方(非 DeepGEMM 路径)不受影响;数值语义上 CUDA 与 Triton kernel 均按 group 求 amax/缩放,PR 未给出逐位对齐声明;列主序 scale 布局需与 DeepGEMM 侧的读取约定一致。

## 关联

直接前序:#4817(sgl-kernel 引入 `sgl_per_token_group_quant_fp8` CUDA kernel)。同区域为 W8A8 block-FP8 量化路径(#32843 同在 fp8_utils.py 的 DeepGEMM/flashinfer 分派);pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
