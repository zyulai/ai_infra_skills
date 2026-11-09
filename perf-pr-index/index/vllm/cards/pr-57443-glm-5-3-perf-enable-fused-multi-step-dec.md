# PR #57443: [GLM 5.3 Perf] Enable fused multi-step decode, 13.3% E2E throughput improvement for concurrency 1

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-10-03 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | speculative-decoding, ready, deepseek, mrv2, glm |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/57443 |

## 一句话结论

为 GLM-5.3 MTP 投机解码启用融合多步 draft decode:元数据只建一次、后续步原地更新,concurrency 1 吞吐 +13.3%。

## 动机

GLM-5.3 走 MTP 投机解码(如 K=5)时,draft 模型的每个 decode 步都要完整重建一次 attention metadata(draft prefill 生成 token #0 后,decode step 1-4 每步 rebuild metadata)。多步 draft 循环里这些重建大多是冗余的:序列长度、slot mapping、调度元数据完全可以由上一步结果增量推导。该开销在小 batch(concurrency 1)时占比最高,直接拖慢单请求延迟。

## 优化点

引入"draft decode 元数据原地更新"机制:attention metadata builder 新增 `supports_draft_decode_metadata_update` 能力标志与 `update_draft_decode_metadata(metadata)` 接口;投机循环第一步完整 build,后续各步只做 in-place 更新——slot_mapping 由 `slot/block_size*block_size + positions%block_size` 就地重算,indexer 路径更新 decode.seq_lens(按 compress_ratio 整除)、decode_lens.fill_(1) 并重算 DeepGEMM `get_paged_mqa_logits_metadata` 的 schedule_metadata,均为显存内原地操作,可被 CUDA graph 捕获。`speculator._build_attn_metadata` 向 builder 传入 input_buffers 的 `positions` 切片以支持增量位置更新。三个相关 builder 落地该能力:DeepseekV32 indexer(`dcp_world_size == 1` 时开启)、KpoolTail(恒开启)、FlashAttnMLASparse(空实现占位,DCP=1 时声明支持)。

## 关键改动

```text
vllm/v1/attention/backends/mla/indexer.py: DeepseekV32/KpoolTail builder 实现 update_draft_decode_metadata
vllm/v1/attention/backends/mla/flashattn_mla_sparse.py: 同上(sparse MLA builder 声明与占位更新)
vllm/v1/worker/gpu/spec_decode/speculator.py: _build_attn_metadata 传入 positions 缓冲
tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py: CUDA graph capture-safe 更新测试
tests/v1/attention/test_kpool_tail_slot_mapping.py: draft slot mapping 增量更新测试
```

核心更新逻辑(indexer 路径,节选):

```python
if self.compress_ratio > 1:
    get_compressed_slot_mapping(..., out=metadata.slot_mapping)
    torch.div(metadata.seq_lens, self.compress_ratio,
              rounding_mode="floor", out=decode.seq_lens.view(-1))
else:
    decode.seq_lens.view(-1).copy_(metadata.seq_lens)
decode.decode_lens.fill_(1)
if current_platform.is_cuda() and has_deep_gemm():
    schedule_metadata = get_paged_mqa_logits_metadata(...)
    decode.schedule_metadata.copy_(schedule_metadata)
```

## 性能声明

PR 正文 guidellm 压测:zai-org/GLM-5.3,TP8,fastsafetensors 加载,`--gpu-memory-utilization 0.88`,prefix caching 开启,speculative-config 为 `mtp,num_speculative_tokens=5,draft_sample_method=greedy,rejection_sample_method=standard`;workload 为 synthetic_text prompt 8000 tok / output 1000 tok,temperature 1.0、top_p 0.95,max_duration 150 s:

| Concurrency | Avg Latency(base→now) | Throughput(base→now) | Avg TTFT(base→now) |
| --- | --- | --- | --- |
| 1 | 6.68 s → 5.90 s(−11.8%) | 150.4 → 170.4 tok/s(+13.3%) | 665.3 → 653.0 ms(−1.8%) |
| 4 | 11.48 s → 11.24 s(−2.1%) | 338.6 → 343.1 tok/s(+1.3%) | 734.7 → 753.5 ms(+2.6%) |
| 16 | 25.61 s → 25.19 s(−1.6%) | 573.8 → 587.4 tok/s(+2.4%) | 1732.1 → 1748.6 ms(+1.0%) |

收益随并发增大而稀释,符合"省调度/CPU 开销、单流最受益"的预期。精度由 unit test 覆盖,正文未给 eval 数字。

## 验证与回归风险

新增 `test_indexer_draft_decode_metadata_update_is_capture_safe`:在 CUDA graph 内执行更新、replay 后校验 decode_seq_lens、schedule_metadata 与 slot_mapping,保证捕获安全;`test_builder_updates_draft_mapping` 验证 KpoolTail 的 slot 增量映射;`test_eagle_draft_attn_metadata`、`test_gpu_autoregressive_speculator` 补 positions 断言。回归风险集中在元数据原地更新路径:DCP(dcp_world_size > 1)明确不支持、回落旧路径;indexer 更新依赖 compress_ratio/DeepGEMM 分支,non-CUDA 或无 DeepGEMM 时跳过 schedule_metadata 更新;声明 `supports_draft_decode_metadata_update=True` 的 builder 若 build 产物缺 positions 会 assert 失败。影响面为使用该批 builder 的 DeepSeek V3.2/GLM-5.3 系 MLA 模型的投机解码。

## 关联

同区域:GLM/DeepSeek MTP 投机解码与 MLA indexer 元数据相关 PR(本批次内无其他同分类卡);PR 标签 deepseek/mrv2/glm 提示其与 DeepseekV32 indexer 基础设施共享。前序:GLM-5.3 模型支持与 DeepSeek V3.2 indexer 引入 PR(正文未给出编号)。
