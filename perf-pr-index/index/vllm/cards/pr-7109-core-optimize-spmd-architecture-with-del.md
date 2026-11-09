# PR #7109: [Core] Optimize SPMD architecture with delta + serialization optimization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @rkooo567 |
| 合并 | 2024-08-19 |
| 分类 | Model-specific Fast Path |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 部分可用,pack 注明另 dropped 14 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/7109 |

## 一句话结论

SPMD worker 改发 SequenceData 增量并用 msgspec 序列化,PP4 llama-8B 吞吐约 +30%,批量推理 +10-15%。

## 动机

Ray SPMD worker 架构下,调度器每步把完整的 `SequenceGroupMetadata`(含每条序列的全部 prompt/output token 列表、block tables 等)序列化后发给所有 worker,再由各 worker 重复做 prepare_inputs。控制面开销在高 TP/PP 或小模型小 batch 场景被显著放大;这是 performance roadmap(#6801)与 #6556、#6241 中明确要解决的问题。

## 优化点

三件事。第一,delta 传输:只有 `SequenceData` 是随时间变化的状态,调度器对非首次 prefill 的请求改发 `SequenceGroupMetadataDelta`,内含每条序列 `get_delta_and_reset()` 产出的新增 token 增量;首次 prefill 仍发完整 metadata(供 worker 建立状态),开关 `SchedulerConfig.send_delta_data` 仅在 `VLLM_USE_RAY_SPMD_WORKER=1` 且 ray 后端时开启。第二,序列化提速:driver 与 worker 之间的 `ExecuteModelRequest`/`SamplerOutput` 改用 msgspec msgpack 编解码(带 array 的 encode/decode hook,token ids 存进 `array(VLLM_TOKEN_ID_ARRAY_TYPE)` 而非 Python list),作者实测 pickle 比 ray 默认序列化至少快 2X,msgspec 又比 pickle 至少快 2X。第三,PP 集成:stage>0 的 worker 同样只接收 delta,中间张量随序列化请求一起传递(`execute_model_spmd` 处理 bytes 元组)。配套把 `SamplingParams`、`LoRARequest`、`PromptAdapterRequest`、`PoolingParams` 从手写 dataclass 改为 msgspec.Struct(omit_defaults/array_like),并移除 `_seq_group_metadata_cache` 对象池。

## 关键改动

```text
vllm/core/scheduler.py: 首次 prefill 发全量,之后发 SequenceGroupMetadataDelta(seq_data delta + block_tables)
vllm/executor/msgspec_utils.py: array 类型 msgspec encode/dec hook(新增)
vllm/executor/ray_gpu_executor.py + ray_utils.py: DAG 边界收发 msgspec bytes,PP stage 间转发序列化请求
vllm/sequence.py 等: token ids 改用 array(VLLM_TOKEN_ID_ARRAY_TYPE);SamplingParams/LoRARequest 等改 msgspec.Struct
vllm/config.py + engine/arg_utils.py: SchedulerConfig.send_delta_data 私有开关
```

关键片段(scheduler 的 delta 分支):

```python
if is_first_prefill or not self.scheduler_config.send_delta_data:
    seq_group_metadata = SequenceGroupMetadata(...)
else:
    seq_data_delta = {}
    for id, data in seq_data.items():
        seq_data_delta[id] = data.get_delta_and_reset()
    seq_group_metadata = SequenceGroupMetadataDelta(
        seq_data_delta, seq_group.request_id, block_tables,
        is_prompt, do_sample=do_sample, token_chunk_size=..., ...)
```

## 性能声明

- TP + 小模型吞吐(opt-125m, tp2):max_num_seqs=256(控制面开销最放大)21.43 → 20.12 req/s,约 5% 回退;max_num_seqs=64(更接近真实负载)13.87 → 14.68 req/s,SPMD 更快。
- PP 在线服务(QPS 10、1000 requests、PP4 + A10G、llama-3-8B、ShareGPT、chunked prefill、OpenAI server):e2e_throughput 2.6 → 3.3 req/s(约 +30%,作者称 chunked prefill 关闭时差距更大);TTFT mean 422.3 → 312.5 ms;ITL mean 207.3 → 155.4 ms;e2e_time 38.7 → 30.2 s。
- PP 批量推理(L4 + PP8,不经 OpenAI server):chunked 562.3 → chunked+SPMD(no nccl)610.7 tokens/s、(nccl)608.6 tokens/s,一般 10-15% 吞吐提升;未 chunked 仅 77.0 tokens/s 作对照。

## 验证与回归风险

作者验证可用场景:chunked prefill、preemption、PP+TP+ray aDAG、seed(已移到 worker);尚不支持 mp 后端。跟进项(WIP):spec decode + SPMD、PP stage>0 跳过 prepare_inputs、LoRA 测试、guided decoding。风险:仅 Ray SPMD + compiled DAG 路径受影响,mp/非 SPMD 不变;msgspec.Struct 替代 dataclass 改变了 SamplingParams 等类的行为面(如 dict=True 为保 cached_property);增量协议要求 worker 状态与调度器严格同步,首次 prefill 判定(`num_computed_tokens == 0`)正确性是关键;测试改动集中在 worker model runner 测试的 array 构造。

## 关联

基于 issue:#6801(performance roadmap)、#6556,精确实现 #6241;同区域:后续 spec decode + SPMD(作者列出的 follow-up)。pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
