# PR #8496: [Bugfix] Fix incorrect llava next feature size calculation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#8496](https://github.com/vllm-project/vllm/pull/8496) |
| 作者 | zyddnys |
| 合并日期 | 2024-09-20 |
| labels | ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
LLaVA-NeXT 特征尺寸计算用整数除法缩放产生舍入误差,多模态 token 数与占位符数不匹配报错;改用浮点 scale_factor 缩放修复。

## 问题现象
旧代码对某些图片算出错误的 feature size,导致报错:`Attempted to assign XXX = XXX multimodal tokens to YYY placeholders`(多模态 token 数与占位符数量不一致)。作者明确给出触发条件:一张 700x840 的图片即可触发。关联 issue 为 #8421,正文未给出更详细的复现日志或影响版本。

## 根因
正文只笼统说明"旧代码产生不正确的 feature size",未披露更细的根因分析。从 diff 看改动机制(由 diff 推断):`_get_llava_next_num_unpadded_features` 中旧代码在把原图尺寸按当前网格尺寸缩放时使用整数除法,如 `new_height = (original_height * current_width) // original_width`、`new_width = (original_width * current_height) // original_height`。当缩放比例不是整数时,先乘后整除会向下取整,得到偏小(偏移)的 new_height/new_width,继而 padding 与最终 `current_height/current_width` 出现偏差,导致 `unpadded_features = current_height * current_width` 与实际多模态 token 数不符。

## 修复方案
只改 `vllm/model_executor/models/llava_next.py` 的 `_get_llava_next_num_unpadded_features`:
1. 变量 `aspect_ratio` 改名 `original_aspect_ratio`(判断分支逻辑不变)。
2. 两个分支改用浮点缩放因子再取整,消除整数除法的舍入误差:
```python
scale_factor = current_width / original_width
new_height = int(original_height * scale_factor)
padding = (current_height - new_height) // 2
current_height -= 2 * padding
```
(另一分支对称地用 `scale_factor = current_height / original_height` 算 `new_width`,`current_width -= 2 * padding`。)
`unpadded_features`、`newline_features` 的后续计算保持不变。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 LLaVA-NeXT(`llava_next.py`)的多模态特征尺寸计算路径,即 `_get_llava_next_num_unpadded_features` 的两个缩放分支。对所有依赖该函数推算 image placeholder 数量的输入尺寸生效;浮点乘法与取整顺序的改变可能改变部分边界尺寸下的取整结果(例如原本恰好整除的场景结果不变,非整除场景结果更精确),若下游有按旧(错误)取整结果硬编码的假设需要留意。

## 关联
- #8421(FIX)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
