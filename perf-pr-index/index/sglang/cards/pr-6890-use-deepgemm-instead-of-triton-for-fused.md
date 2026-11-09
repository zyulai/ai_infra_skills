# PR #6890: Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @fzyzcjy |
| 合并 | 2025-06-05 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready-to-merge |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/6890 |

## 一句话结论

放宽 DeepGEMM W8A8 block FP8 路径的形状检查(N 维 %64),让 `fused_qkv_a_proj_with_mqa` 走 DeepGEMM 而非 Triton,该 kernel 实测快 3x 以上。

## 动机

`fused_qkv_a_proj_with_mqa` 的输出维度不满足 DeepGEMM 路径原有的 `weight.shape[0] % 128 == 0` 形状检查,于是回退到较慢的 Triton 实现。作者实测该 kernel:Triton 41 微秒,DeepGEMM 12 微秒,在他场景下超过 3x 提速,值得让这条路径进入 DeepGEMM。

## 优化点

改动极小,只有一处形状检查:`deepgemm_w8a8_block_fp8_linear_with_fallback` 中 `shape_supported` 从 `weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0` 放宽为 `weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0`。N 维(K 行方向)从要求 128 对齐降到 64 对齐,使 MLA 模型的 fused qkv_a 投影权重也能命中 DeepGEMM 分支;不满足时仍回退 Triton。diff 注释中留下 TODO 指向 PR 内的 review comment,提示后续需要更健壮的形状检查。机制上这是把既有 DeepGEMM fallback 路径的适用范围扩大,没有新 kernel。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_utils.py: deepgemm_w8a8_block_fp8_linear_with_fallback 的 shape_supported 从 N%128 放宽到 N%64
```

```python
shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0
if not (shape_supported and dtype_supported):
    # fall back to triton
```

## 性能声明

PR 正文仅给出 kernel 级两点测量:triton 41 微秒 vs DeepGEMM 12 微秒(即 3x 以上加速),并注明是"in my scenario";模型、硬件、精度、workload 均未注明。无端到端数据。PR 最初标记 "do not merge now, I will test and check more first"(后已合并)。

## 验证与回归风险

PR 未引入测试;checklist 各项(格式化、单测、文档、benchmark)在合并时均未勾选。风险面:放宽的形状约束影响所有命中 `deepgemm_w8a8_block_fp8_linear_with_fallback` 的调用方——N 维在 [64, 128) 区间非 128 对齐的权重现在会走 DeepGEMM 而不是 Triton,数值路径改变;DeepGEMM 对这些形状的正确性依赖其内部约束,PR 通过 review comment 跟进(见 diff 中 TODO 链接)。dtype 侧不变(仅 bf16 输出受支持,其余仍回退)。

## 关联

同区域:DeepGEMM W8A8 block FP8 linear 与 Triton fallback 的选择逻辑;MLA 模型的 `fused_qkv_a_proj_with_mqa` 投影;pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
