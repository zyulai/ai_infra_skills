# PR #13162: Fix nan in global scaling factor for large scale nvfp4 EP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13162](https://github.com/sgl-project/sglang/pull/13162) |
| 作者 | wenscarl |
| 合并日期 | 2025-11-12 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
大规模 nvfp4 EP 下 `w13_input_scale` 依赖 `logical_to_all_physical_map` 造成漏行空洞,`torch.empty` 初始化出 NaN;改为读取全部物理专家。

## 问题现象
正文未描述运行期现象,问题即标题所示:大规模 nvfp4 EP 下 global scaling factor 出现 NaN。正文称本 PR 属于 #12866 的一部分(其中相关改动「reverted in this PR」,即由本 PR 回退后重做)。正文给出的关键数据:`w13_input_scale` 形状为 [288, ...],而当 `logical_to_all_physical_map` 为

```
[
[286],
[287],
[2],
[3],
...,
[254],
[255]
]
```

时第 0、1 行缺失;由于 `w13_input_scale` 由 `torch.empty` 初始化,对应位置的值是未定义的。

## 根因
正文披露:`w13[2]_input_scale` 不应依赖 `logical_to_all_physical_map`。当该 map 缺少部分 logical expert id 对应的行时,按其索引填充 `w13_input_scale` 会留下空洞,而 `torch.empty` 初始化使这些位置为未定义值,从而产生 NaN。修复思路是不管 logical expert id 是什么,都读取全部物理专家的 input scale。

## 修复方案
正文未填 Modifications 小节,以下为 diff 显示的改动:
- `python/sglang/srt/eplb/expert_location.py`:`logical_to_all_physical()` 新增 `require_global_experts: bool = False` 参数,为 True 时直接返回 `arange(0, num_physical_experts)`(全部物理专家);`_compute_logical_to_all_physical_map` 中 nearest expert 的写回由「仅当 nearest 不在映射中才替换首元素」改为「`nearest_expert != -1` 时直接置为 `[nearest_expert]`」。
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的 `weight_loader`:读取参数上的 `_sglang_require_global_experts` 属性并传入 `logical_to_all_physical`。

```diff
+        if require_global_experts:
+            num_physical_experts = self.logical_to_all_physical_map_cpu[layer_id].shape[
+                -1
+            ]
+            return list(torch.arange(0, num_physical_experts))
```

## 影响与回归风险
- 回归面:`logical_to_all_physical()` 为 eplb 公共方法,新参数默认 False,不传时调用行为不变(由 diff 推断);`_compute_logical_to_all_physical_map` 的 nearest-expert 写回逻辑变化会影响 eplb 重映射结果;`weight_loader` 对带 `_sglang_require_global_experts` 标记的量化参数改为加载全部物理专家的 scale。

## 关联
- [#12866](https://github.com/sgl-project/sglang/pull/12866)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
