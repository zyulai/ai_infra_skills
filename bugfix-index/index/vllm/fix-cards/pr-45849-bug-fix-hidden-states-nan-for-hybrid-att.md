# PR #45849: [BUG] fix hidden states nan for hybrid attention models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45849](https://github.com/vllm-project/vllm/pull/45849) |
| 作者 | shanjiaz |
| 合并日期 | 2026-06-17 |
| labels | bug, kv-connector, ready, v1 |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
hybrid 模型上 ExampleHiddenStatesConnector 硬编码 group 0 且 HiddenStateCacheSpec 未被清零,读到未初始化 NaN 缓存,两处修复。

## 问题现象
混合注意力模型(如 Qwen3.5,同时含 linear_attention 与 full_attention 层)使用 `ExampleHiddenStatesConnector` 时返回 NaN hidden states。复现脚本 `python repro_nan_hidden_states.py`(与 transformers 参考实现对比):修复前 vLLM 输出 shape=[205, 2560]、nan_tokens=205/205(100%),FAIL;修复后 nan_tokens=0/205(0%),PASS。

## 根因
正文披露(编号列出两处,自称 "Fixed three main issues"):
1. 连接器的 `_hs_group_idx` 被硬编码为 `0`。对有多个 KV cache group 的 hybrid 模型,hidden-states 所在 group 不一定是 group 0;这导致 `request_finished_all_groups` 把错误 group 的 block IDs 传给 reader,reader 进而从未初始化的缓存位置提取 hidden states。修复特意在 `__init__` 扫描 `kv_cache_groups`(而非 `register_kv_caches`),因为 scheduler 侧连接器会用到它但从不调用 `register_kv_caches`。
2. `SingleTypeKVCacheManager` 中 `type(spec) in (...)` 只匹配精确类型,`HiddenStateCacheSpec`(`MLAAttentionSpec` 的子类)被排除在 block 清零之外,新分配的 block 可能残留 stale NaN/garbage。

## 修复方案
- `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py`:在 `__init__` 遍历 `self._kv_cache_config.kv_cache_groups`,找到 layer_names 含 `cache_only_layers` 的 group 作为 hidden-states group:
```python
self._hs_group_idx: int = 0
if self._kv_cache_config is not None:
    for i, group in enumerate(self._kv_cache_config.kv_cache_groups):
        if any("cache_only_layers" in n for n in group.layer_names):
            self._hs_group_idx = i
            break
```
- `vllm/v1/core/single_type_kv_cache_manager.py`:在 `allocate_external_computed_blocks` 与 `allocate_new_blocks` 两处的类型判断元组中加入 `HiddenStateCacheSpec`,使其新分配/外部已计算 block 的 block_id 进入 `new_block_ids` 参与清零。

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:使用 ExampleHiddenStatesConnector 的多 KV group 模型(_hs_group_idx 不再固定为 0,单 group 模型仍解析为 0,行为不变);所有含 HiddenStateCacheSpec 的 SingleTypeKVCacheManager 分配路径(新增类型进入清零列表,只多清零、不减);正文称修复三处但 diff 仅体现上述两个文件,第三处改动在展示的 diff 中不可见。

## 关联
- [#45734](https://github.com/vllm-project/vllm/issues/45734)(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
