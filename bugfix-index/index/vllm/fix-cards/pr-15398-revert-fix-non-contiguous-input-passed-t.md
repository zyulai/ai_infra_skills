# PR #15398: Revert "Fix non-contiguous input passed to Marlin kernel (#15319)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15398](https://github.com/vllm-project/vllm/pull/15398) |
| 作者 | tlrmchlsmth |
| 合并日期 | 2025-03-25 |
| labels | ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#14658 已支持 noncontiguous Marlin 内核,#15319 的 x.contiguous() 修复不再需要,整体 revert。

## 问题现象
正文未描述具体现象(仅说明 revert 理由:该修复不再需要)。

## 根因
正文披露:#14658 已为 Marlin 内核添加 noncontiguous 支持,#15319 的修复因此不再需要,故 revert 其 commit `d20e261199d3e4bffe9ee0e495ba7f7454229f11`。

## 修复方案
删除 `vllm/model_executor/layers/quantization/kernels/mixed_precision/marlin.py` 中 `apply_weights` 开头的三行:
```python
# marlin requires contiguous memory layout
# prefix caching may cause x to be non-contiguous
x = x.contiguous()  # no-op if already contiguous
```
即 `apply_weights` 不再把输入 `x` 强制变为连续内存。

## 影响与回归风险
- 回归面:Marlin `apply_weights` 的输入处理路径;prefix caching 等场景产生的非连续输入不再被 `contiguous()` 拷贝规整,改依赖 #14658 引入的内核 noncontiguous 支持,省去一次可能的显存拷贝。

## 关联
Revert #15319;依赖 #14658

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
