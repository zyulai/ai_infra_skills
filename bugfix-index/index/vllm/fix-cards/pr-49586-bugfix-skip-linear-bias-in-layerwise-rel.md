# PR #49586: [Bugfix] Skip linear bias in layerwise reload to avoid corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49586](https://github.com/vllm-project/vllm/pull/49586) |
| 作者 | li-jinpeng |
| 合并日期 | 2026-07-24 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
在线量化 + layerwise reload 路径会把已加载的线性层 bias 用未初始化内存重分配而永久损坏;把 `bias` 加入 SKIP_TENSORS 跳过处理。

## 问题现象
带 bias 的 Linear 层经在线量化 + layerwise-reload 路径(如在线 FP8)加载后,`bias` 中是未初始化内存而非 checkpoint 值,且永久不可恢复。正文实测:加载 Qwen3 vision tower(`visual.blocks.*` 的 qkv/proj/mlp)时,修复前 110 个 vision-tower bias 全部与 bf16 checkpoint 不匹配(allclose 失败,值为未初始化/零);修复后 110 个全部精确匹配。追踪 `visual.blocks.0.attn.proj` 确认:`materialize_layer()` 前后 bias 指针改变、absmax 从 3.578 降到 0。

## 根因
正文披露:Linear 基类在 `create_weights()` **之后**才创建 `bias`,而在线量化在 `create_weights()` 内部就调用 `initialize_online_processing()`,即发生在 bias 存在之前。于是:
1. `load_numel_total = get_layer_size(layer)` 只统计了 `weight`;
2. bias 的 weight loader 未被包进 `online_process_loader`,因而从不缓冲进 `info.loaded_weights`。
weight 一加载完,`load_numel >= load_numel_total` 成立、`_layerwise_process()` 触发;`materialize_layer()` 用 `empty_strided` 重新分配该层所有未被跳过的张量——包括已加载、非 meta 的 bias——真实值被未初始化内存替换;又因 bias 不在 `loaded_weights` 里,replay 循环不会写回,损坏成为永久性的。

## 修复方案
`vllm/model_executor/model_loader/reload/meta.py` 的 `SKIP_TENSORS` 集合加入 `"bias"`:
```python
     "e_score_correction_bias",
+    # Built after create_weights(), so it is not tracked by the layerwise-reload
+    # trigger and would be re-materialized into uninitialized memory. Skip it.
+    "bias",
```
正文论证:bias 从不参与量化(FP8 只量化 weight,bias 保持 bf16、在 GEMM 后相加),体积极小,且本来就在普通路径 eager 加载,因此永远不应被移到 meta 或被 layerwise-reload 重分配。

## 影响与回归风险
- 回归面:layerwise reload 的张量跳过集合;所有带 bias 的 Linear 层不再进入 meta/重分配流程,加载顺序与普通路径一致。不带 bias 的层与其他 SKIP_TENSORS 条目行为不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
