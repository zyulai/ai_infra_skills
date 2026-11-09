# PR #49793: [Spec Decode][Perf] Fuse the MTP trailing all-reduce; local-argmax draft tokens

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhou9402 |
| 合并 | 2026-08-15 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, v1, mrv2 |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 仅改动 `vllm/models/deepseek_v32/nvidia/mtp.py` 一个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/49793 |

## 一句话结论

DeepSeek-V3.2/GLM-5.2 MTP draft 路径两处通信优化：尾随 all-reduce 融入最终 RMSNorm、draft token 改 vocab 分片本地 argmax 免全词表 all-gather，c=64 吞吐 +13.6%。

## 动机

MTP draft 每步结束时先做一次显式 `tensor_model_parallel_all_reduce`（MoE 输出未归约），再单独过 `shared_head.norm`——这与主模型在层边界已做的 all-reduce+RMSNorm 融合不一致，多出一次独立通信；同时 `compute_logits` 为了取 greedy draft token 做全词表 all-gather，c=64 时每步约 64 × 151k × 2B ≈ 19 MB、每个接受 token 重复五次。proposer 侧已通过 `use_local_argmax_reduction` 探测该能力，缺模型侧实现。

## 优化点

两处改动（仅 DSv3.2 NVIDIA MTP 模型文件）。其一，非 sequence-parallel 路径：删除显式 all-reduce 与独立的 `shared_head.norm` 调用，改为 `fused_allreduce_rms_norm(hidden_states, residual, self.shared_head.norm)`——把 MoE 未归约输出的 all-reduce 融合进最终 RMSNorm（residual-add 只算一次，结果同时供两个元组位置使用，兼容 V2 speculator 与 legacy proposer 的元组协议）；sequence-parallel 路径保持 `norm + sp_all_gather` 不变。其二，新增 `get_top_tokens(hidden_states, spec_step_idx)`（模型与外层包装两个层级）：经 `logits_processor.get_top_tokens` 在各 rank 的 vocab 分片上做 argmax 得到 greedy draft token id，跳过 `compute_logits` 的全词表 all-gather——方法名由 proposer 的 `use_local_argmax_reduction` 探测协议固定。PR 说明早期版本携带的 `index_share_for_mtp_iteration` 门控与 V2 speculator 生命周期 hook 已在 main 落地，本版已剔除。

## 关键改动

```text
vllm/models/deepseek_v32/nvidia/mtp.py: 非 SP 路径改 fused_allreduce_rms_norm；新增 get_top_tokens 本地 argmax draft
```

```python
if is_sequence_parallel:
    hidden_states, _ = self.shared_head.norm(hidden_states, residual)
    hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
else:
    # The MoE output is left un-reduced; fuse its all-reduce into the
    # final norm, as the main model does at layer boundaries.
    hidden_states, _ = fused_allreduce_rms_norm(
        hidden_states, residual, self.shared_head.norm
    )
```

## 性能声明

测试条件：2xGB300、TP8 跨 2 节点，`nvidia/GLM-5.2-NVFP4`，MTP=5，`VLLM_USE_V2_MODEL_RUNNER=1`，fp8 KV cache，无 prefix caching；baseline 为同树回退成显式 all-reduce+RMSNorm 的版本。Batch 1（8192 in/1024 out，c=1，每 arm 三次）：无可测变化（三 arm 输出 tok/s 区间完全重叠，median ITL 9.08–9.12ms）——c=1 时省掉的 all-gather 只有一行 vocab、融合的 all-reduce 是小张量。并发 64（1024 in/1024 out、256 prompts，单对 A/B）：baseline 2923.2 tok/s / acceptance 4.86 / median ITL 31.83ms → +both 3321.1 tok/s（+13.6%）/ acceptance 4.69 / median ITL 31.32ms；按 acceptance length 归一化的 steps/s 为 +17.8%（接受率低 3.5% 排除投机运气因素）。作者明示注意事项：c=64 为单对 A/B（复测因集群竞争丢失），跨节点方差大，两表数字不可跨表比较。精度：GSM8K 5-shot 全 1319 题，flexible-extract 0.9439±0.0063→0.9431±0.0064、strict-match 0.9424±0.0064→0.9409±0.0065（stderr 内不变；融合 kernel fp32 累加与本地 argmax 的 tie-breaking 差异原则上影响输出）。

## 验证与回归风险

测试：diff 未含新增单测；验证依赖上述 A/B 基准与 lm_eval GSM8K。风险面：只影响 DeepSeek-V3.2/GLM-5.2 MTP 的 NVIDIA 路径（`deepseek_v32/nvidia/mtp.py`）；非 SP 路径输出语义改变（fp32 累加顺序、argmax tie-breaking）——精度验证显示 stderr 内无变化；`get_top_tokens` 依赖 proposer 侧 `use_local_argmax_reduction` 探测协议，未探测时走原 `compute_logits`；SP 路径与 ROCm 路径未改动。

## 关联

同区域：主模型层边界的 fused all-reduce+RMSNorm（#34299 使其默认开启，索引内）；`use_local_argmax_reduction` 探测为 proposer 侧既有机制（本 PR 补模型侧）。PR 正文未列具体关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
