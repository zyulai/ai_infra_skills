# PR #50904: [GLM Perf] DSv32/glm use skip topk for MTP case, 2.0x kernel performance improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-08-06 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未见「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/50904 |

## 一句话结论

MTP 复用 step 0 的 Top-K buffer 跳过 Indexer 重算，H200 上内核 168.33us→84.22us（2.00x）。

## 动机

本 PR 是 issue #46654 的一部分，目标是去掉 MTP（Multi-Token Prediction）draft step 中的冗余计算。正文指出，此前 `proposer.set_skip_topk(True)` 虽设置了跳过标志，但 eager `_fused_attention` 只判断 `indexer != None`，仍令 `has_indexer=True`，`fused_norm_rope` 收到 `index_k=None` 后触发重算（recompute），Top-K 并未真正被跳过。

## 优化点

在 `_fused_attention` 的多个分支把判断由 `self.indexer is not None` 改为 `self.indexer is not None and not self.skip_topk`。当 `skip_topk=True` 时置 `has_indexer=False`，跳过 Indexer 的 Q/K 处理、跳过 `sparse_attn_indexer` 与 Top-K 计算，直接复用 step 0 已产生的 Top-K buffer，然后继续正常的 MLA attention。改动落在 DSv32/GLM 的 attention 路径及 ROCm 侧对应逻辑，属于 speculative decoding 的 draft-step 计算裁剪，不改变 MLA 主体。

## 关键改动

- `vllm/models/deepseek_v32/attention.py`：核心改动，`_fused_attention` 三处分支增加 `and not self.skip_topk`，使 has_indexer 为假并跳过 sparse_attn_indexer / Top-K。
- `vllm/models/deepseek_v32/amd/rocm.py`：ROCm 侧对应的 skip_topk 适配。
- 单测：正文称 "Acc covered in current unit test"，性能由正文所附 AI 生成脚本验证。

```text
-        if self.indexer is not None:
+        if self.indexer is not None and not self.skip_topk:
             has_indexer = True
             indexer_k_norm_w = self.indexer.k_norm.weight
             indexer_k_norm_bias = self.indexer.k_norm.bias
```

## 性能声明

正文标题称 "2.0x kernel performance improvement"，regex 检出 "speedup: 2.00x"。所附 benchmark 脚本输出（GPU: NVIDIA H200；case: batch=1, context=8192, TP=8, BF16 query；GLM-5.2 TP=8 维度，TOPK=2048，`VLLM_USE_DEEP_GEMM=1`，WARMUP=100、ITERATIONS=1000、TRIALS=7；checkpoint GLM-5.2-NVFP4）：before (compute Indexer) 168.33 us、after (reuse Top-K) 84.22 us、saved 84.11 us、speedup 2.00x、latency reduction 50.0%。这是 Indexer 计算的内核级 microbenchmark，正文未给出端到端 serving 吞吐数据；模型精度 PR 未注明测试条件（仅称单测覆盖 accuracy）。

## 验证与回归风险

验证：正文称精度由现有单元测试覆盖，性能由上述 AI 生成脚本测得。可能受影响路径：DSv32/GLM 的 MTP draft step、eager `_fused_attention`、sparse_attn_indexer / Top-K 计算、ROCm 侧 attention；风险在于复用 step 0 的 Top-K buffer 是否被所有 draft step 正确沿用，正文未列出具体回归测试文件名。

## 关联

正文标注 "Part of https://github.com/vllm-project/vllm/issues/46654"（issue 编号而非 PR 编号），未给出同分类其它 PR 编号或前序/后续 PR，按 PR 编号口径写「无」。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
