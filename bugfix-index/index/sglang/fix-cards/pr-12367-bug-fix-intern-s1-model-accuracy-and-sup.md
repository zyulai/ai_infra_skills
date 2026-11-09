# PR #12367: [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12367](https://github.com/sgl-project/sglang/pull/12367) |
| 作者 | hhaAndroid |
| 合并日期 | 2025-11-03 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
intern-s1 用错 pixel_shuffle v1 版本致图像转置、精度下降;删除 ps_version 分支,放宽 load_weights 校验以支持 RL 增量更新,并让 /generate 支持 input_ids。

## 问题现象
正文列出三点,未附日志或复现步骤:
1. `intern-s1` 使用的 `pixel_shuffle` 版本不正确,需移除 `ps_version` 参数。
2. `intern-s1` 与 `internv1` 的图像处理模型不支持通过 generation 接口传入 `input_ids`。
3. RL 场景下权重更新是增量的,interns1 与 internvl 的 `weight update` 相关代码需要修复。

被删除的旧代码警告文本(见 diff)描述了 v1 版本的症状:"In ps_version 'v1', the height and width have not been swapped back, which results in a transposed image.",即 ps_version 为 v1 时宽高未换回、图像被转置。

## 根因
正文披露:`pixel_shuffle` 的 `ps_version` 逻辑导致 intern-s1 走了错误版本(精度问题);多模态处理器未支持 generation 接口传 `input_ids`;RL 增量权重更新路径有缺陷。具体机制方面:diff 显示 `load_weights` 末尾存在「有未加载参数即抛 RuntimeError」的严格校验,与「RL 增量更新会失败」对应,该对应关系为(由 diff 推断);多模态处理器中 `input_text.replace` 使用的源文本改为 `base_output.input_text`(由 diff 推断:原先的 `input_text` 缺少 input_ids 对应内容导致不支持)。

## 修复方案
- `python/sglang/srt/models/interns1.py`:删除 `self.ps_version` 与相关日志;`pixel_shuffle` 无条件执行 `x = x.permute(0, 2, 1, 3).contiguous()`(不再按 v1 跳过):
```diff
-        if self.ps_version == "v1":
-            logger.warn(...)
-        else:
-            x = x.permute(0, 2, 1, 3).contiguous()
+        x = x.permute(0, 2, 1, 3).contiguous()
```
- `interns1.py` 与 `internvl.py` 的 `load_weights`:删除 `loaded_params` 跟踪与「未加载参数」`RuntimeError` 校验(internvl 中还删除了 FP8 `weight_scale` 等量化参数的白名单过滤),使增量权重更新不再被误报中断。
- `python/sglang/srt/multimodal/processors/internvl.py`:`input_text` 改由 `base_output.input_text` 替换 IMG_CONTEXT 占位符,支持经 generation 接口传入 `input_ids`。
- `EntryClass` 由列表改为直接类引用(清理项)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:intern-s1/internvl 视觉特征提取(pixel_shuffle 行为固定为 permute 版本)、两模型 `load_weights` 不再对缺失权重报错(缺权重将静默通过,排错信息减少)、internvl 多模态预处理输入文本来源改变;`EntryClass` 写法变化。不影响其他模型。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
