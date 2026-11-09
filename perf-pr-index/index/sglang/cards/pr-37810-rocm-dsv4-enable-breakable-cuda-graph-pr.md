# PR #37810: [ROCm][DSV4] Enable breakable CUDA graph prefill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AMD-yanfeiwang |
| 合并 | 2026-09-16 |
| 分类 | Speculative Decoding |
| Label | amd, deepseek, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,可见部分涉及 8 个文件,另有 1 个文件因超长未展示 |
| 链接 | https://github.com/sgl-project/sglang/pull/37810 |

## 一句话结论

为 ROCm DeepSeek-V4 HIP radix 后端启用 breakable CUDA graph prefill,隔离 prefill 场景 TTFT 最高降 49.9%。

## 动机

breakable CUDA graph(BCG)prefill 此前未覆盖 ROCm 的 DeepSeek-V4 HIP radix 后端(通用 ROCm BCG 支持见 #27833,CUDA 侧 DSV4 BCG metadata contract 见 #25195)。prefill 路径 kernel launch 与 metadata 构建开销在低并发、短输出场景占比高,graph replay 可消除;但 DSV4 的 unified-KV/indexer/compressor/FP4 工作区元数据都是逐 batch 变化的,需要解决 capture/replay 间的元数据刷新问题。

## 优化点

核心是"captured-metadata"模式:HIP radix 后端置 `use_captured_forward_metadata_for_breakable_cuda_graph=True`,为 `UnifiedKvMetadata`/`DSV4AttnMetadata`/`DSV4Metadata` 新增 `refresh_for_breakable_cuda_graph_replay_()`,每次 replay 前把当前 batch 的动态字段(swa/hca/csa indices 与 indptr、page_table、topk lengths、c4/c128 out_loc、compressor 元数据、fp4 k_write/q_positions 等)in-place `copy_` 进 capture 时的张量,保持图内指针地址不变。capture 以 `MAX_SEQ_LEN_FOR_CAPTURE` 覆盖 max_seq_len。消除隐式 D2H 同步:`expand_prefill_causally` 的 padding 用 `torch.cat + expand` 替代 `.item()`;host 侧 `extend_seq_lens_cpu` 求和恰好等于 num_tokens 时直接证明 exact_num_tokens;显式传 `repeat_interleave(output_size=...)` 与 `extend_start_loc`;`_attach_unified_kv_prefill_meta` 将元数据 pad 到 capture bucket,padded 行 `pf_final_pos=unified_swa_window` 使 SWA store 的 guard 跳过。准入策略:MIXED batch 在 DP attention 下保持 eager(新后端标志 `prefer_eager_mixed_prefill_under_dp_attention`,经 prefill graph vote 的 `is_mixed` 生效),纯 TP 与其他后端保持原 replay 策略;有 running decode 请求时零 graph 记录。BCG 保持 opt-in:`--cuda-graph-backend-prefill breakable`。

## 关键改动

```text
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: BCG capture/replay 元数据构建与刷新
python/sglang/srt/layers/attention/base_attn_backend.py: 新增 captured-metadata / MIXED-eager 后端开关
python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py: is_mixed 准入策略接入
python/sglang/srt/managers/scheduler_components/dp_attn.py: prefill graph vote 携带 is_mixed
python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py: padding 去除 .item() D2H 同步
test/registered/amd/test_dsv4_hip_bcg_metadata.py: 新增 BCG 元数据单测
```

```python
def prepare_forward_metadata_for_breakable_cuda_graph_replay(self, capture_metadata, forward_batch, ...):
    replay_metadata = self._build_forward_metadata(replay_batch, max_seq_len_override=self.MAX_SEQ_LEN_FOR_CAPTURE, use_prefill_cuda_graph=True)
    capture_metadata.refresh_for_breakable_cuda_graph_replay_(replay_metadata)
    self.forward_metadata = capture_metadata
```

## 性能声明

8 x MI355X,TP8 + DP8 attention,DP LM head,`unified_kv_triton`,无投机解码,3 repeats(seeds 42/43/44),仅 prefill 后端不同(eager vs BCG):

- 1K→1K C4-C64:输出吞吐 delta -1.04% ~ +1.65%,paired median 多在 ±1% 内,基本 throughput-neutral
- 最高准入点(1K, C4):TTFT 282.634 → 188.345 ms(-33.361%),吞吐 +0.990%
- 隔离 prefill(纯 TP8,较早 image):`1024->64` C1 下 TTFT 146.219 → 73.194 ms(-49.942%),输出吞吐 +9.584%,E2E -8.579%;receive coalescing + [1K,2K,4K,8K] buckets、graph-token 准入 100% 时 `1024->1` C1 request-throughput +74.298%,TTFT 164.864 → 92.969 ms
- AgentX 生产负载(DSpark + HiCache,1x MI355X TP8):吞吐 +0.00% ~ +1.57%;TTFT p50 -13.96% / -2.98% / -10.15%(C16/32/48);BCG graph 活跃 709/1058/1243 次
- Ready-state 显存:BCG 较 eager +0.943 GiB/GPU
- 作者结论:保持 opt-in,无显著 aggregate 吞吐回归,也不作普遍吞吐收益声明

## 验证与回归风险

新增 `test/registered/amd/test_dsv4_hip_bcg_metadata.py`(capture/replay 元数据、DP admission、地址稳定性、underfilled bucket、no-D2H guard):20 tests + 2 subtests passed;42/42 性能样本精确 token 总量;冷/热全 rank parity 64/64(eager/BCG 文本与 token ID 完全一致,DP0-DP7);3/3 server lifecycle 干净退出。AgentX 验证依赖 #39007(`record_stream_for_v2_verify`),且 ROCm 需显式 `SGLANG_ENABLE_WAR_BARRIER=1`。风险:refresh 字段清单必须与 kernel 实际读取的元数据完全一致,漏一项即静默错配;MIXED-eager 策略仅限 ROCm DSV4;+0.943 GiB/GPU 显存开销。CI:Base/Extra 通过,AMD ROCm 10 失败。

## 关联

References:#25195(CUDA DSV4 BCG metadata contract)、#27833(generic ROCm BCG 支持)、#27380(ROCm DSV4 unified-KV backend);依赖:#39007(AgentX 验证路径)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
