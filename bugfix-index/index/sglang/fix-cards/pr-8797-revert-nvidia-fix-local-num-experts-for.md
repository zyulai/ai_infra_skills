# PR #8797: Revert "[NVIDIA]Fix local_num_experts for EP (#8779)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8797](https://github.com/sgl-project/sglang/pull/8797) |
| 作者 | zhyncs |
| 合并日期 | 2025-08-05 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论

整体回滚 #8779:恢复 `create_weights` 原有的单参数 `num_experts=self.num_local_experts` 传参,删除新增的 `num_local_experts` 形参,消除其引入的回归。

## 问题现象

正文未描述具体现象。正文仅一句 "This reverts commit b01eeb80f8406cba569af5deb40f394293f9950d.",Motivation/Modifications 等模板段均未填写。

## 根因

正文未披露根因。(由 diff 推断)被回滚的 #8779 曾把 `fused_moe_triton/layer.py` 中 `create_weights` 的调用从单参数 `num_experts=self.num_local_experts` 改为同时传 `num_experts=self.num_experts`(全局专家数)与新增的 `num_local_experts=self.num_local_experts`,并在 `modelopt_quant.py` 中新增 `num_local_experts` 形参;该改动疑似引发 MoE 权重创建/量化路径的回归,故被整体撤销。

## 修复方案

按 revert 提交恢复两处改动(共 2 个文件):

`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`:

```diff
-            num_experts=self.num_experts,
-            num_local_experts=self.num_local_experts,
+            num_experts=self.num_local_experts,
```

`python/sglang/srt/layers/quantization/modelopt_quant.py` 的 `create_weights`:删除 `num_local_experts: int` 形参,恢复 `layer.num_local_experts = num_experts` 的赋值方式,即回到 #8779 之前的签名与语义。

## 影响与回归风险

- 影响窗口:含 #8779 改动(提交 b01eeb80)的版本受影响;本 PR 合并后恢复原行为(依据正文 revert 声明与被回滚提交号)。
- 回归面:改动触及 MoE 权重创建的公共调用点(`fused_moe_triton/layer.py`)与 ModelOpt 量化实现(`modelopt_quant.py`)。恢复后所有经该路径创建权重的模型回到 #8779 之前行为(`num_experts` 形参重新接收本地专家数);#8779 原本针对的 NVIDIA EP 场景如确有需求,需要以新方案重做。

## 关联

- [#8779](https://github.com/sgl-project/sglang/pull/8779)(标题中引用的被回滚 PR;正文未另行列出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
