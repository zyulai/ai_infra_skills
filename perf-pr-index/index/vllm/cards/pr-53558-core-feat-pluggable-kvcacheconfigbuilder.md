# PR #53558: [Core][Feat] Pluggable KVCacheConfigBuilder for platform/model-specific KV cache planning

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wangxiyuan |
| 合并 | 2026-10-16 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | ready, kv-connector, nvidia, mrv2, dflash, kv-cache-manager |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 "1 file(s) dropped from diff" |
| 链接 | https://github.com/vllm-project/vllm/pull/53558 |

## 一句话结论

把 KV cache 分组/放置从 core 单一 heuristic 解耦为可插拔 builder，新模型与平台无需再改 core。

## 动机

Follow-up to #42309（RFC #62100）。vLLM 用 core 中单一 heuristic（`get_kv_cache_groups()`）决定每个模型的 KV cache grouping。新 hybrid 模型、drafters、cache precisions 与 platforms 不断需要该 heuristic 给不出的 plan，于是每个新模型都要改 core，vllm-ascend 甚至 monkey-patch core planning 来使用自己的 DSV4 plan。本 PR 加入 experimental 可插拔 `KVCacheConfigBuilder`：core 保留单一 planning flow 并检查每个 plan，platform 选 builder 决定 grouping 与 placement，model 可返回自己的 groups。接口可能变化，尤其 #61452 改动 KV cache group 数据结构时。

## 优化点

`KVCacheConfigBuilder`（`vllm/v1/core/kv_cache_config_builder.py`）只做两个决定：`get_kv_cache_groups(vllm_config, kv_cache_spec)` 与 `get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)`；platform 用 `Platform.get_kv_cache_config_builder_cls` 选类，默认 `DefaultKVCacheConfigBuilder`。Core owns the planning flow：`get_kv_cache_configs` 是 core 函数而非 builder 方法，负责 merge specs、group、check、project onto PP stages、apply block-count override、auto-fit and admission、place、converge ranks。Bytes per block 来自 builder 的 placement——core 放一个 block 读其单一 device backing，有自己 placement 的 platform 不再单独算内存；有多个 device allocation 的 placement 覆盖私有过渡性的 `_get_pool_bytes_per_block`（直到 worker-driven allocation #42449）。Model-owned grouping：model class 可实现 `SupportsKVCacheGrouping.get_kv_cache_groups`（classmethod），优先级为 override grouping 的 platform builder（仅通过 `super()` 到 model hook）、model hook、默认 heuristic。每个 grouping 都在 planning path 与两个 profiling paths 被检查（每层恰在一个 group）；profiling 改为直接传 `num_blocks` 而非临时设置 `num_gpu_blocks_override`。Bug fixes：PP projection 保留部分投影 group 的 `host_resident`、`role`、`enable_kv_transfer`；`num_gpu_blocks_override` 不再膨胀 one-block probe（原给 `500 * C` 而非 `C` bytes per block）；HiSparse 保留直接 bytes 计算，model hook 与 HiSparse 组合报清晰错误。

## 关键改动

```text
vllm/v1/core/kv_cache_config_builder.py: 新增 KVCacheConfigBuilder 抽象与 get_kv_cache_config_builder 解析
vllm/v1/core/kv_cache_planning.py: DefaultKVCacheConfigBuilder、core-owned planning flow、profiling 直接传 num_blocks
vllm/platforms/interface.py: Platform.get_kv_cache_config_builder_cls hook
vllm/model_executor/models/interfaces.py 与 vllm/model_executor/models/registry.py: SupportsKVCacheGrouping 接口与解析
tests/v1/core/test_kv_cache_config_builder.py: builder 解析、优先级、HiSparse 拒绝的回归测试
```

关键代码（来自 `tests/v1/core/test_kv_cache_config_builder.py` diff）：

```python
def test_public_interface_declares_two_placement_hooks(self):
    # get_kv_cache_configs lives in kv_cache_planning and is Core-owned;
    # the builder only declares the placement hooks.
    assert KVCacheConfigBuilder.__abstractmethods__ == {
        "get_kv_cache_groups",
        "get_kv_cache_config_from_groups",
    }
    assert issubclass(DefaultKVCacheConfigBuilder, KVCacheConfigBuilder)
```

## 性能声明

PR 正文未给出定量性能数据（吞吐/延迟/加速比）。Test Result 仅给出 `83 passed (head 6283b0aca3, GB200 node)`；Users 一节提到 wangxiyuan/vllm#2 的 model hooks 使 Kimi-K3 + DSpark 的分组由 20 组减为 5 组（分组数量变化，非性能测量）。PR 未注明测试条件（硬件仅注明 GB200 node，无模型/精度/workload 的 benchmark 条件）。

## 验证与回归风险

Test Plan：`pytest tests/v1/core/test_kv_cache_config_builder.py tests/v1/core/test_kv_cache_planning.py`；Test Result：`83 passed (head 6283b0aca3, GB200 node)`。正文说明使用 AI assistance (Claude Code)、作者逐行复核，并列出三位 Co-Authored-By。风险面：KV cache planning 与两条 profiling 路径、PP projection 的部分投影 group、`num_gpu_blocks_override` 的 one-block probe、HiSparse 的 bytes 计算、platform-specific placement（vllm-ascend#18359 把 Ascend plans 迁入 `AscendKVCacheConfigBuilder`），以及 #61452 未来改动 KV cache group 数据结构带来的接口变化。

## 关联

#62100（RFC）、#42309（前序 follow-up）、#61452（KV cache group 数据结构变更）、#42449（worker-driven allocation）、#58638/#58207/#58380（改进 drafters 默认 heuristic，正文明确为互补而非重复）；外部：vllm-ascend#18359、wangxiyuan/vllm#2（stacked，model hooks for Qwen3.8-Flash-Next 与 Kimi-K3）。均为正文明确出现的编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
