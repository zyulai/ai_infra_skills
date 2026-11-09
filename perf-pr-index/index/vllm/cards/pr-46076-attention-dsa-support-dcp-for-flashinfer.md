# PR #46076: [Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-07-01 |
| 分类 | Attention / Kernel |
| Label | documentation, ready, v1, nvidia |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面);diff 因超长截断,另有 8 个文件未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/46076 |

## 一句话结论

为 FLASHINFER_MLA_SPARSE(DSA sparse indexer)打通 decode context parallel,GB200 上 decode 吞吐提升约 62%。

## 动机

GLM 5.2 使用 DSA(DeepSeek Sparse Attention)稀疏 indexer 配 FLASHINFER_MLA_SPARSE 后端,但该后端此前不支持 DCP(decode context parallel):超长上下文(1M token 级)请求在纯 TP 下 KV cache 容量与 decode 并发受限,无法把 decode 上下文切到多机。

## 优化点

为 sparse MLA decode 路径补齐 DCP 所需的完整链路:各 rank 对本地 KV 分片做 top-k,再经一次 packed all-gather 合并出全局 top-k——新增 CuteDSL kernel `stable_topk_from_gathered_candidates_cutedsl`(以 score 降序、全局 token id 升序的稳定序选择,配合 Triton pack kernel `_pack_dcp_topk_candidates_triton_kernel` 打包 score+global_id 候选对);`triton_filter_and_convert_dcp_index` 在 kernel 内做 rank 属主过滤、全局/本地索引换算与 compaction(atomic 槽位分配,把 rank 拥有的物理槽位压实到前缀);prefill 侧 `build_prefill_chunk_metadata` 按 `cp_kv_cache_interleave_size` 将 `cu_seqlen_ks/ke` 因果边界本地化;多 rank 输出经 LSE merge(`correct_attn_out` 处理空行 NaN)合并。surplus top-k 槽位统一以 -1 填充并在 merge 中视为 invalid。

## 关键改动

```text
vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py: CuteDSL 稳定 top-k 选择 + Triton 候选打包(新增)
vllm/v1/attention/backends/mla/sparse_utils.py: triton_filter_and_convert_dcp_index 属主过滤与 compaction
tests/v1/attention/test_indexer_dcp_localize.py: 新增 951 行 DCP 本地化/合并等价性测试
docs/design/attention_backends.md: FLASHINFER_MLA_SPARSE 的 DCP 列改为支持
```

## 性能声明

GB200,`vllm serve lukealonso/GLM-5.2-NVFP4 --kv-cache-dtype fp8_e4m3 -tp 4 [-dcp 4]`,concurrency 256 decode:dcp4 generation throughput 4071.3 tokens/s(Running 241)vs 纯 tp4 2508.7 tokens/s(Running 63),约 +62%;但 "dcp prefill speed is only 1/2 of tp, so the e2e throughput is not ideal"。KV 容量(TP8 对照):DCP4+TP8 为 10,853,376 tokens / 1,048,576-token 请求最大并发 10.35x,纯 TP8 为 2,765,568 tokens / 2.64x。质量:GLM-5.2-FP8 TP8+DCP4 上 lm_eval gsm8k 5-shot flexible 0.9340 / strict 0.9333。

## 验证与回归风险

新增 `test_indexer_dcp_localize.py` 覆盖 local seq lens 本地化、DCP top-k 合并与非 DCP 参考实现的 out/lse 等价(world/interleave 参数化)、surplus -1 padding 不变量、kernel 内 compaction 与参考集合等价、short-context decode 端到端链路。风险:sparse indexer 的 DCP 路径依赖 CuteDSL(`has_cutedsl()` 检查),top-k 集合确定性依赖 radix kernel 的排列不变约定。

## 关联

同区域:PR #37421(DeepSeek-V3.2 DSA decode persistent TopK scheduler)、PR #47327(sparse MLA 短序列 dense MHA 路径),均作用于 DSA/sparse MLA 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
