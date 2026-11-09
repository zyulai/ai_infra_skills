# PR #30419: [NIXL][BUG FIX] Fix both failing issue and accuracy issue with nixl + host_buffer on CUDA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30419](https://github.com/vllm-project/vllm/pull/30419) |
| 作者 | xuechendi |
| 合并日期 | 2025-12-18 |
| labels | kv-connector, nvidia, ready, v1 |
| 热度 | 👍 1 · 💬 13 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

NIXL + CPU host buffer 跑 PD 分离时,chunked prefill 的后续分块未注册进 nixl metadata 导致失败与精度劣化;在 running 队列处理时补注册并保留未完成请求。

## 问题现象

在 CUDA 上用 nixl + host_buffer 跑 PD(prefill/decode 分离)时出现两类问题:一是运行失败(修 #29665 之后引入的 bug),二是精度问题(见 issue #30358)。实测 Qwen3-0.6B 精度从约 0.3 恢复到 0.4109。测试命令:`PREFILL_BLOCK_SIZE=16 DECODE_BLOCK_SIZE=16 bash tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh --kv_buffer_device cpu`。

## 根因

正文披露:scheduler 调用 `schedule` 时,只有全新请求才会走 `self.connector.update_state_after_alloc`;当一个 prefill 请求被切成多个小请求(chunked prefill)时,`update_state_after_alloc` 只把第一个请求的部分 block_ids 注册进 nixl_metadata,后续分块被调度时不再更新,导致 nixl metadata 中块信息不完整——既造成失败,也造成精度劣化。

## 修复方案

- 把 `yield_req_data` 从 `offloading_connector.py` 上移到 `kv_connector/utils.py`,同时遍历 `scheduled_new_reqs` 与 `scheduled_cached_reqs`(含 resumed 标记)。
- `nixl_connector.py`:`_reqs_need_save` 改为 `dict[ReqId, Request]`(不再预先存 block_ids);`build_connector_meta` 里改用 `yield_req_data` 遍历,对已登记保存的请求取 `new_block_id_groups[0]` 注册进 metadata,并按 `num_computed_tokens + num_scheduled_tokens < num_prompt_tokens` 判断是否为部分 prefill——非部分才从 `_reqs_need_save` 弹出,部分 prefill 保留请求直到所有块都被调度。
- 删除 `build_connector_meta` 末尾对 `_reqs_need_save` 的整体 `clear()`(避免清掉尚未注册完整的请求)。
- `request_finished` 中请求中止(如 partial prefill 的 abort)时 `self._reqs_need_save.pop(request.request_id, None)` 清理。

## 影响与回归风险

- 影响窗口:正文说明运行失败问题是 #29665 之后引入;精度问题对应 issue #30358。未给出具体版本号。
- 回归面:改动作用于 NIXL KV connector 的 scheduler 侧元数据构建路径(`build_connector_meta`、`update_state_after_alloc`、`request_finished`),仅影响使用 nixl connector(尤其 host_buffer/PD 分离)的部署;`yield_req_data` 迁移同时被 `offloading_connector.py` 复用,其行为等价但引用路径变化;`_reqs_need_save` 生命周期变化(由整体 clear 改为按请求 pop)若某请求永远不被调度也未中止,可能滞留该 dict。

## 关联

- #30420(依赖的前置 PR)
- #29665(失败问题引入点)
- #30358(精度问题 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
