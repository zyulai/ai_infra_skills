# RFC #15254: [RFC]: Better support for weight updating while waking up from sleep mode for RLHF

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15254](https://github.com/vllm-project/vllm/issues/15254) |
| 作者 | erictang000 |
| 创建日期 | 2025-03-20 |
| 状态 | ⚪ closed (2026-02-06) |
| labels | RFC, stale |
| 热度 | 👍 18 · 💬 14 |
| 分类 | Feature / Ecosystem |

## 一句话结论
为 RLHF 的 sleep/wake_up 流程改进内存时序:支持分阶段唤醒与就地权重更新,避免唤醒时 OOM。

## 提案动机
在 verl 等 RLHF 框架中用 sleep mode + `wake_up()` 时会出现 OOM(verl issue #302):verl 的 fsdp_vllm sharding manager 要唤醒 vllm 引擎并更新权重时,同一张 GPU 上同时存在旧权重、已废弃的 KV cache 与新权重三份内存,大模型场景直接 OOM。理论上可先把新权重 offload 到 CPU 再加载规避,但多数 RLHF 库希望用 NCCL 让权重留在 GPU 上传输,应尽量避免 CPU offload。

## 提出的方案
提出一个临时改进加一个后续里程碑:
- 临时改进:支持分阶段唤醒——`wake_up(tags=("weights",))` 与 `wake_up(tags=("kv_cache",))` 分开调用,先只重分配权重内存、执行 `llm.inplace_update_weights(new_weights)` 更新并释放临时内存后,再唤醒 KV cache 内存;在 V1 引擎中还需新增类似 `llm.inplace_update_weights(new_weights)` 的 `collective_rpc` 调用来就地更新权重(V0 中可直接访问 `model.load_weights`);
- 后续里程碑:提供 `llm.wake_up_with_weights(new_weights)` 这类 API,实现完全细粒度的就地权重更新——在 `CumemAllocator` 中新增端点,交错进行内存重映射、拷贝新权重、拷贝完成后释放新权重内存;可把峰值 GPU 内存占用压到最低,但可能需要修改模型代码以支持逐层 weight loader。

## 优化点与收益
避免 RLHF 权重同步时唤醒 vllm 引擎引发的 OOM(作者举例的场景:在 A100-80G 上更新 32B 这类较大模型),从而让 verl 等 RLHF 库能设置更高的 GPU 内存利用率(gpu_memory_utilization);后续的就地更新方案还可把峰值显存占用降到最低。上述为定性收益与场景条件,正文未给出量化数字。

## 现状与落地
已关闭(2026-02-06),labels 含 stale。正文处于方案讨论阶段(作者 CC 了 @youkaichao @comaniac),正文未记录方案是否被实现或被替代,落地情况需查关联 PR。

## 关联
- https://github.com/volcengine/verl/tree/main (verl RLHF 框架)
- https://github.com/volcengine/verl/issues/302 (唤醒时 OOM 的具体 issue)
- https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/fsdp_vllm.py#L94 (问题所在的 fsdp_vllm sharding manager)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
