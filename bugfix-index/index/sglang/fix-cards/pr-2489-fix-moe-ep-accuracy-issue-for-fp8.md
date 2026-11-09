# PR #2489: fix moe-ep accuracy issue for fp8

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2489](https://github.com/sgl-project/sglang/pull/2489) |
| 作者 | xiaobochen123 |
| 合并日期 | 2024-12-16 |
| labels | (none) |
| 热度 | 👍 2 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
EP MoE 加载 FP8 模型时精度出错;修复为在 process_weights_after_loading 中对 w13_weight_scale 沿 dim=1 取 max 压缩,得到静态量化期望的单组激活 scale。

## 问题现象
MoE EP(expert parallel)模式加载 FP8 模型时出现精度问题(正文原话:「fix moe ep bug, when load fp8 model」)。正文关联 issue #2482,测试模型为 `neuralmagic/DeepSeek-Coder-V2-Instruct-FP8`,修复后评测结果:Accuracy 0.932、Invalid 0.000、Latency 243.824 s、Output throughput 1027.530 token/s。正文未描述修复前的具体数值或报错现象。

## 根因
正文未披露根因(正文只有一句动机与链接的 issue)。(由 diff 推断)在 `process_weights_after_loading` 的静态量化分支(校验「QuantConfig has static quantization, but found activation scales are None」之后),`layer.w13_weight_scale` 原始形状包含 dim=1 维度(即 gate/up 两个分支各自成组的 scale),而该静态量化路径后续期望的是沿 dim=1 已约简的单组激活 scale;未约简时尺寸与后续计算不匹配,导致 MoE EP + FP8 精度错误。修复取 `torch.max(w13_weight_scale, dim=1).values`,把每组两个 scale 合并为较大值作为激活 scale。

## 修复方案
修改 `python/sglang/srt/layers/ep_moe/layer.py` 的 `process_weights_after_loading`:在静态量化校验分支里,对 `layer.w13_weight_scale` 沿 dim=1 取 max 后重建 Parameter,`requires_grad=False`:

```python
+                layer.w13_weight_scale = torch.nn.Parameter(
+                    torch.max(layer.w13_weight_scale, dim=1).values,
+                    requires_grad=False,
+                )
```

仅一处改动,单文件。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 EP MoE FP8 静态量化路径的权重后处理:w13_weight_scale 尺寸沿 dim=1 被约简(数值上取 gate/up 两 scale 的较大者,属于保守近似),此后所有依赖该参数形状的 apply/量化计算走新形状;非静态量化或非 FP8 的路径不进入该分支,行为不变。正文未给修复前精度对照,回归面评估有限。

## 关联
- [#2482](https://github.com/sgl-project/sglang/issues/2482)(相关 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
