# PR #28942: [Bugfix]  Fix precision corruption when shared_experts_stream=None

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28942](https://github.com/vllm-project/vllm/pull/28942) |
| 作者 | Jasen2201 |
| 合并日期 | 2025-11-19 |
| labels | bug, ready, rocm |
| 热度 | 👍 1 · 💬 7 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
MoE 共享专家流为 None 时,共享专家在路由专家矩阵乘之后执行,而矩阵乘会原地改写 `hidden_states`,精度被污染;改为在矩阵乘之前先算共享专家。

## 问题现象
`shared_experts_stream=None`(无独立共享专家流)时输出精度损坏。作者在 ROCm(gfx942)用 DeepSeek-R1、TP=8、`VLLM_ROCM_USE_AITER=1` 复现:prompt「The capital of China」温度 0,修复前输出为乱码文本「 Beijingisnownsgrmetically Beijing BeijingBeijingBeijing…」;修复后输出正确「 is Beijing, and the capital of Russia is Moscow. Both China and Russia have」。

## 根因
正文披露(Purpose):`shared_experts_stream=None` 时共享专家未走独立流路径,原本在路由专家 kernel(矩阵乘)之后才执行;而矩阵乘可能会原地修改(mutate in-place)`hidden_states`,共享专家读到的是被污染后的输入,导致精度错误。

## 修复方案
改 `vllm/model_executor/layers/fused_moe/layer.py` 的 `forward_impl`:在路由专家矩阵乘之前、当存在独立共享专家且未使用共享专家流时先算共享输出;删除矩阵乘后 `else` 分支中的共享专家调用:

```python
# Run shared experts before matrix multiply.
# because matrix multiply maybe modify the hidden_states.
if has_separate_shared_experts and not use_shared_experts_stream:
    assert self.shared_experts is not None
    shared_output = self.shared_experts(hidden_states)

# Matrix multiply.
final_hidden_states = self.quant_method.apply(...)
```

另将 `vllm/utils/torch_utils.py` 的 `aux_stream()` 判定从 `is_cuda()` 改为 `is_cuda_alike()`,使 ROCm 等类 CUDA 平台也能创建辅助流(注释同步更新)。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:`fused_moe/layer.py` 影响所有带独立共享专家的 MoE 层在「无共享专家流」配置下的 forward 顺序——共享专家改在 EP dispatch 后、矩阵乘前执行;走独立流的路径(`use_shared_experts_stream=True`)仍用 `hidden_states_clone` 在辅助流上计算,不变。`aux_stream()` 改动使 ROCm 平台开始启用共享专家辅助流重叠,属于行为变化点。验证仅覆盖 ROCm gfx942 + DeepSeek-R1 TP=8 单场景,其他平台/模型未在正文列出。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
