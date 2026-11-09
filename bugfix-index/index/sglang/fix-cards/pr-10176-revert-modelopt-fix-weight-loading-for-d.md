# PR #10176: Revert "[ModelOpt] Fix Weight Loading for DSR1-FP4 Quantization (#9712)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10176](https://github.com/sgl-project/sglang/pull/10176) |
| 作者 | zhyncs |
| 合并日期 | 2025-09-08 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
整体回滚 #9712 对 ModelOpt DSR1-FP4 权重加载的修复,该修复关联 issue #10175,先撤销再议。

## 问题现象
正文未描述具体现象,仅给出关联 issue #10175 并执行回滚;Motivation/Modifications 等小节均为空模板占位。

## 根因
正文未披露根因,仅声明 "This reverts commit fcd72bd100b5bdad4b304e2c76b82e657edf9502"(即 #9712 的改动)并链接 issue #10175,可推断 #9712 的修复本身引入了问题或未达预期(正文未展开细节)。
(由 diff 推断)被回滚的 #9712 改动包含两处:`linear.py` 的 `weight_loader` 将简单断言换成带尺寸详情报错信息的断言;`modelopt_quant.py` 的 `is_layer_excluded` 增加了「exclude 模式最后一段包含于 prefix 最后一段即视为命中」的模糊匹配(用于 fused_qkv_a_proj_with_mqa 等融合模块),本次 revert 将两者恢复原状。

## 修复方案
原样回滚 commit fcd72bd100b5bdad4b304e2c76b82e657edf9502:
- `python/sglang/srt/layers/linear.py`:`weight_loader` 的断言恢复为 `assert param.size() == loaded_weight.size()`(去掉自定义报错文案)。
- `python/sglang/srt/layers/quantization/modelopt_quant.py`:`is_layer_excluded` 删除新增的模糊匹配逻辑:
  ```diff
  -            # Check if the last part of the excluded pattern is contained in the last part of the prefix
  -            pattern_last_part = pattern.split(".")[-1]
  -            prefix_last_part = prefix.split(".")[-1]
  -            if pattern_last_part in prefix_last_part:
  -                return True
  ```

## 影响与回归风险
- 影响窗口:正文未说明;按 revert 对象,#9712 合并之后、本 PR 之前的提交受影响。
- 回归面:ModelOpt(ModelOpt FP4 量化)权重加载的 exclude 模块匹配与 `weight_loader` 断言路径;回滚后 #9712 原本要解决的 DSR1-FP4 加载问题可能复现(原 issue 的效果由 #10175 跟踪)。同时 `is_layer_excluded` 行为恢复对其他 ModelOpt 模型的权重加载也回到旧逻辑。

## 关联
- https://github.com/sgl-project/sglang/issues/10175
- https://github.com/sgl-project/sglang/pull/9712 (被 revert 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
