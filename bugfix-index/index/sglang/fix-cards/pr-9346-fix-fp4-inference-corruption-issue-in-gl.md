# PR #9346: Fix FP4 inference corruption issue in glm4.5-air model

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9346](https://github.com/sgl-project/sglang/pull/9346) |
| 作者 | Azure-Tang |
| 合并日期 | 2025-08-21 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
FP4 量化算子用 `torch.empty` 分配 scale 张量,padding 区随机值在 nvfp4 映射中变 NaN 污染 GEMM;有 padding 时改用 `torch.zeros`。

## 问题现象
glm4.5-air 模型使用 FP4 量化时,推理输出可能包含损坏/乱码文本(关联 #8814)。glm4.5 第一层 MLP 的 intermediate size 为 10944,tp=2 时触发 padding,因此该问题总在 glm4.5 第一层被触发;其他层或其他模型中,只要不规则形状导致量化 padding,同样可能发生。修复后 glm4.5-air FP4 推理不再产生损坏输出,多个 benchmark 均以正确结果跑通。

## 根因
正文披露:sglang 的量化算子此前用 `torch.empty` 创建张量;padding 时 `empty` 会引入随机值,这些随机值随后在映射到 nvfp4 格式时被转换成 NaN;NaN 进入 glm4.5 第一层 MLP 的矩阵并传播到后续 GEMM 计算中。

## 修复方案
`sgl-kernel/python/sgl_kernel/gemm.py` 两处,在确实发生 padding 时改用显式初始化的张量:
- `scaled_fp4_quant`:`output_scale` 在 `rounded_n > scale_n`(发生 padding)时改用 `torch.zeros`,否则保持 `torch.empty`;
- `scaled_fp4_experts_quant`:`output_scales` 在 `padded_k > scales_k` 时同理改用 `torch.zeros`。

```python
# padded part should be zeroed out
if rounded_n > scale_n:
    output_scale = torch.zeros(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )
else:
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )
```

保证 padding 区域被清零,随机值不再被转成 NaN 向下游传播。由 @Azure-Tang 与 @qiuqiu10 共同完成。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 scale 输出张量的分配方式;无 padding 的形状仍走原 `torch.empty` 路径,不引入额外清零开销;受影响的只有触发 padding 的不规则形状(如 tp=2 下的 glm4.5 首层)。

## 关联
#8814

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
