# PR #26563: [Bugfix][Multi Modal] Fix incorrect Molmo image processing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26563](https://github.com/vllm-project/vllm/pull/26563) |
| 作者 | sangho-vision |
| 合并日期 | 2025-10-11 |
| labels | ready |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
#12966 把 Molmo 的 image_input_idx 重排序删掉换成布尔掩码,patch 位置错乱导致 pointing 输出错误,恢复按 image_input_idx 重排并修正 token 计数。

## 问题现象
Molmo 模型输出不正确,典型表现为 pointing 任务错位(issue #26451 报告 Molmo pointing 结果不正确;#26518 讨论确认)。用 `allenai/Molmo-7B-D-0924` 对「Point to the car.」测试,修复后输出 `<point x="69.0" y="48.6" alt="car">car</point>`,坐标与图中汽车位置正确对齐。

## 根因
正文披露:#12966 删除了把图像特征映射回对应 patch token 位置的 `image_input_idx` 重排序,换成布尔掩码 `feat_is_patch`;该掩码只表示每个图像特征是否被用作多模态输入序列中的 patch token,不表示每个有效 patch 特征应放到哪里,缺少重建原始 patch 顺序所需的逐特征索引,导致图像特征与 patch token 位置的对齐被破坏。另外 `MolmoProcessingInfo.get_num_image_tokens` 未计入 image start/end token 以及列分隔 token,token 计数也不正确。

## 修复方案
`vllm/model_executor/models/molmo.py`:
- `MolmoImageInputs` 的 `feat_is_patch` 字段替换回 `image_input_idx`(索引张量,映射图像特征到对应 patch token),`_get_mm_fields_config`、`_parse_and_validate_image_input`、处理器输出同步更名。
- `_process_image_input` 恢复重排序:对每个 crop,过滤 `img_idx >= 0` 的有效特征后按 `torch.argsort(valid_img_idx)` 排序:

```python
is_valid = img_idx >= 0
valid_img_idx = img_idx[is_valid]
order = torch.argsort(valid_img_idx)
results.append(feats[is_valid][order])
```

- `get_num_image_tokens` 修正计数: `extra = 2 + (image_token_length_w + 1) * image_token_length_h`,`joint = 2 + ((ncols + 1) // pooling_size + 1) * ((nrows + 1) // pooling_size)`,计入 start/end 与列分隔 token。

## 影响与回归风险
- 影响窗口:#12966 合入之后、本 PR 之前的 Molmo 部署(pointing/定位类任务输出错位)。
- 回归面:Molmo 多模态输入 schema 字段名变更(`feat_is_patch` → `image_input_idx`)及特征重排逻辑;token 计数变大(start/end/列 token 计入)会影响最大图像 token 预算与 dummy 数据估算;正文指出 #26451 亦修复重排序但缺 token 计数修正,本 PR 为二者的并集,合入时需注意与之的冲突处理。

## 关联
FIX #26518;讨论见 #26518;并发修复 PR #26451;根因引入 #12966

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
