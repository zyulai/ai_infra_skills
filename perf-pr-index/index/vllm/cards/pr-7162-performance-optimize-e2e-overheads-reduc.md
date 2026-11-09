# PR #7162: [Performance] Optimize e2e overheads: Reduce python allocations

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2024-08-09 |
| 分类 | Model-specific Fast Path |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 部分可用,pack 注明另 dropped 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/7162 |

## 一句话结论

调度/采样路径用对象池复用 Python 对象并加快路径,1xH100 Llama3-8B e2e 吞吐 +24%。

## 动机

每个 scheduler 迭代都会大量分配又释放 Python 对象:`InterDataForSeqGroup` 字段众多且大多动态创建,`SequenceGroupMetadata`、`ScheduledSequenceGroup`、`SequenceGroupToSample`、block table id 列表等每步重建;非 beam search 时每请求仍按多序列写法走 list comprehension。这些短命对象让 CPU 侧 e2e 开销(GC、分配)成为小 batch/decode 密集负载的瓶颈之一。

## 优化点

引入通用 `PyObjectCache`(初始 128 个、不足时倍增,`reset()` 供下一迭代复用),在四处落地:模型 runner builder 的 `InterDataForSeqGroup`(新增 `simple_reinit`,单序列时原地清空复用)、scheduler 的 `SequenceGroupMetadata`(clear seq_data/block_tables 后 `__init__` 复用)与 `SchedulerRunningOutputs`/`ScheduledSequenceGroup`、采样侧按"每 group 序列数"分桶的 `SamplingMetadataCache`(`SequenceGroupToSample` 复用)。算法上给"单序列 per group"加快速路径:`SequenceGroup.is_single_seq` 使 `get_seqs/num_seqs/get_unfinished_seqs` 等不再建列表;block table 状态改用自维护 `BlockTable` 类,块与 block_ids 缓存同步维护,`get_block_table` 直接返回 `ids()`。杂项:`compute_slot_mapping` 对"decode 下一个 token"的常见情形跳过 pad 循环;prefix caching 关闭时跳过 `get_common_computed_block_ids`;`n_blocks` 用整数上取整替代 `math.ceil`;`_schedule_default` 复用列表避免拼接;`blocks_to_copy` 仅在非空时 extend;flash_attn 滑窗切片在 window=0 时直接取全表。

## 关键改动

```text
vllm/utils.py: PyObjectCache 对象池(get_object/reset/自动扩容)
vllm/core/scheduler.py: 三类对象接入缓存,schedule() 复用 SequenceGroupMetadata,跳过非 prefix-caching 的 computed block 查询
vllm/model_executor/sampling_metadata.py: SamplingMetadataCache 按 num_seqs 分桶复用 SequenceGroupToSample
vllm/block.py + core/block_manager_v1.py: BlockTable 类缓存 block_ids,单序列分配免 copy
vllm/sequence.py: SequenceGroup.is_single_seq 快速路径
vllm/worker/model_runner.py: InterDataForSeqGroup reinit 复用
```

关键片段(decode 单 token 的 slot mapping 快路径):

```python
if start_idx == 0 and (seq_len - context_len) == 1:
    # Optimization for common-case of decoding next token
    add_slot(seq_len - 1)
else:
    slot_mapping.extend([PAD_SLOT_ID] * max(0, start_idx - context_len))
    for i in range(max(start_idx, context_len), seq_len):
        add_slot(i)
```

## 性能声明

meta-llama/Meta-Llama-3.1-8B-Instruct,单卡 H100,TP1,`benchmark_throughput.py --input-len 512 --output-len 256 --num-prompts 1000`:

- main(2024-08-05):19.48 requests/s,14962.18 tokens/s。
- 本 PR:24.32 requests/s,18123.51 tokens/s。e2e 吞吐快 24%。

## 验证与回归风险

pack 的 diff 与正文未列新增测试文件(仅对既有代码路径改动,pack 声明 dropped 1 个文件)。回归风险:对象池复用要求使用者严格在迭代边界 reset,任何遗漏会把脏状态带入下一批(`SequenceGroupMetadata.seq_data/block_tables` 靠调用方 clear);`InterDataForSeqGroup.reinit` 假设 seq_ids 长度不变;`SequenceGroupToSample` 复用依赖 prompt_logprob/sample_indices 每次 clear;`RequestOutput` 直接引用 `seq.data._output_token_ids` 绕过 tuple 化,输出侧别名语义变化;`get_seqs` 返回共享列表,调用方若原地修改会影响 group 状态;beam search(fork、copy)路径保留原 copy 逻辑。

## 关联

同区域:PR #7109(SPMD delta 传输,同样针对控制面开销,并移除了本 PR 引入的 `_seq_group_metadata_cache` 改为发 delta);v1 引擎后续以增量协议方式替代此类对象池。pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
