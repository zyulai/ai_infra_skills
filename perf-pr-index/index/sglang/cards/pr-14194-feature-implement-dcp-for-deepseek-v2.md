# PR #14194: [feature] implement dcp for deepseek_v2

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @staugust |
| 合并 | 2026-06-25 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | high priority, deepseek, run-ci, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用(另有 14 个文件因超长未包含在 pack 中) |
| 链接 | https://github.com/sgl-project/sglang/pull/14194 |

## 一句话结论

为 DeepSeek-V2 实现 Decode-Context Parallelism(DCP),把 KV cache 按 rank 切分以在 8×H20 TP8 下支持更长上下文。

## 动机

长上下文场景下单卡 KV cache 容量成为瓶颈。本 PR 是实现 #12196(在 8×H20、TP8 下支持更长上下文)的第一步:将 decode 阶段的 KV cache 沿 DCP 组切分,用多卡聚合容量换长上下文。初始版本仅支持 flashinfer attention backend,兼容 chunked-prefill 与 decode cuda graph,不支持 radix cache、PD 分离与 MTP;2025-12-04 更新后已支持 prefix cache。

## 优化点

1. **计算侧三步通信**:(a) 每个 DCP rank 用 `num_tp_local_q_heads × dcp_size` 个 head 对部分 KV 计算 absorbed_q 的 attn_output,随后 all_gather 使 DCP 组内每个 TP rank 持有完整 absorbed_q;(b) 跨 DCP 组收集各 rank 的 lse,用 Triton kernel `_correct_attn_cp_out_kernel` 按 online softmax(数值稳定版)计算缩放因子并校正 attn_output;(c) 对校正后的 attn_output 做 reduce_scatter,每个 TP rank 保留 `tp_local_q_heads` head、完整 KV 的最终输出,与纯 TP 语义一致。
2. **虚拟容量扩展**:`TokenToKVPool` 的 `real_kv_size` 不变,新增 `DCPTokenToKVPoolAllocator` 对 ScheduleBatch 表现出 `real_kv_size × dcp_world_size` 的可分配容量。
3. **索引对齐策略**:为每个 token 分配一个 KV buffer index,并以 `dcp_world_size × original_page_size` 为对齐单元,保证 `token_idx % dcp_world_size == out_cache_loc % dcp_world_size`,使 `out_cache_loc` 到本 rank 实际 KV 位置的映射仅需按余数过滤、再整除换算,读写逻辑极简。
4. **平台扩展与配套**:DCP 此前仅限 HIP,本 PR 放开到 CUDA;`parallel_state.py` 新增 `reduce_scatter_along_dim()`(含 symmetric memory 支持)与 `get_dcp_rank/world_size`;`dcp_size > 1` 时设 `NCCL_GRAPH_MIXING_SUPPORT=0`(对称内核性能优化,引 NCCL issue);scheduler 的 max_total_num_tokens 观测值乘以 dcp_size。
5. **prefill 路径**:通过 `DecodeContextParallelMetadata` 携带 dcp_kv_buffer/indptr/indices,extend 时 all-gather 前缀 KV 并按 query 顺序重排,chunk-extend 亦有专门 all-gather 路径。

## 关键改动

```text
python/sglang/srt/layers/utils/dcp_utils.py: 新增 724 行核心模块:LSE 校正
  Triton kernel 与 CPTritonContext(防重编译)、DCPTokenToKVPoolAllocator
  相关 metadata 准备、decode/prefill/chunk-extend 的 KV all-gather、
  filter_dcp_local_kv_indices、plan_dcp_decode_metadata
python/sglang/srt/distributed/parallel_state.py: reduce_scatter_along_dim();
  DCP 平台检查放宽到 CUDA;dcp group 辅助接口
python/sglang/srt/layers/attention/flashinfer_mla_backend.py: decode 返回
  lse(return_lse=dcp_enabled());prefill 使用 dcp_kv_buffer/indices;
  cuda graph metadata 的 local kv lens 更新
python/sglang/srt/mem_cache/common.py + scheduler.py: DCP 分配器接入与
  虚拟容量(max_total_num_tokens × dcp_size)
python/sglang/srt/entrypoints/engine.py: dcp_size>1 时设置
  NCCL_GRAPH_MIXING_SUPPORT=0
```

关键片段(diff 摘录,out_cache_loc 到本 rank KV 的映射):

```python
filtered_local_indices = forward_batch.out_cache_loc[
    (forward_batch.out_cache_loc % dcp_world_size) == dcp_rank
]
actual_locations = filtered_local_indices // dcp_world_size
kv_buffer[actual_locations] = kv[filtered_local_indices]
```

## 性能声明

PR 正文未给出加速比类性能数字(仅精度与吞吐对比,GSM8K 200 题,`benchmark/gsm8k/bench_sglang.py`):

- **DCP8 + chunked-prefill + radix cache**:Accuracy 0.975,Invalid 0.000,Latency 31.077s,Output throughput 609.384 tok/s。
- **TP8 基线**(sglang commit `e8ba5a668c3f16bb2e54328971bb00a8295c9e01`):Accuracy 0.970,Invalid 0.000,Latency 25.061s,Output throughput 751.645 tok/s。

即 DCP8 在精度持平(0.975 vs 0.970)的前提下,该 workload 吞吐低于纯 TP8 基线(609 vs 752 tok/s);PR 未注明测试所用模型与硬件细节(动机段指向 8×H20)。

## 验证与回归风险

- 精度验证为上述 GSM8K 对比;Checklist 中 unit tests / documentation 未勾选。CI 状态(pack 记录):Base 与 Extra 均通过。
- 限制(PR 正文):初期仅 flashinfer attention backend;不支持 PD disaggregation 与 MTP;FlashMLA 路径对 FP8 KV cache 显式 assert 不支持 DCP。
- 回归面:`parallel_state.py` 的 reduce_scatter 新路径与 symmetric memory 交互(代码内自注 `reduce_scatter_tensor` 对非连续输入可能有 bug 需 contiguous);DCP 分配器对 ScheduleBatch 的容量语义变化;decode 输出从单 tensor 变为可能返回 `(out, lse)` tuple,影响上层调用约定(代码内 TODO 注明需统一)。
- `update_local_kv_lens_for_dcp` 修改 cuda graph 的 kv_len 数组,与 CUDA graph 捕获/回放路径强耦合。

## 关联

#12196(总体目标:8×H20 TP8 长上下文支持,本 PR 为第一步)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
