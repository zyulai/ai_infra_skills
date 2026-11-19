# PR #65656: [Perf][DSV4.1] Accelerate candidate prefill and decode with CuTe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chaunceyjiang |
| 合并 | 2026-11-17 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ready, ci/build, deepseek, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 新增 vllm/models/deepseek_v41/nvidia/ops/candidate_indexer.py、candidate_paged_logits.py、tests/models/deepseek_v41/nvidia/test_candidate_indexer.py,并改 vllm/models/deepseek_v41/attention.py、vllm/config/attention.py、vllm/model_executor/models/config.py 等 |
| 链接 | https://github.com/vllm-project/vllm/pull/65656 |

## 一句话结论

为 DeepSeek V4.1 candidate indexer 在 SM90 上新增模型本地 CuTe DSL 打分器:直接用 TMA/WGMMA 从分页 FP8 cache 读候选键,decode 与合格的长上下文 prefill 共享一条路径,512K–约 1M 工作负载端到端输出吞吐 +3.2%、ITL P50 最高 −28.4%。

## 动机

V4.1 两级索引器此前在 SM100 走 DeepGEMM sparse MQA-logits、ROCm 走 aiter,SM90 只有稠密 logits 打分;#62463(yeshihai)在 Triton 里实现了 candidate-decode 打分,#62000 先 gather 候选键再打分。本 PR 换用 TMA/WGMMA 的 CuTe DSL 实现、把 prefill 与 decode 统一进同一条打分路径,并扩展到长上下文 prefill,受 #61830 启发。

## 优化点

1. 直接对既有候选池打分,保留模型候选选择策略;从 paged FP8 cache 直接读键产生 FP32 logits。
2. 候选准备跨 consumer layer 共享:row bounds 与物理页映射每步准备一次,logits scratch 由 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 限界。
3. prefill 选择规则:最长压缩 prefill 上下文 ≥ 2×(候选池容量与最大压缩 query chunk 中较大者)才走 candidate 打分(V4.1-Flash query chunk 至多 16K,即 32K 压缩上下文阈值);其余 prefill batch 整批走 dense 打分以保留键复用。
4. 经既有 `indexer_sparse_logits` 配置启用:支持的配置默认开,显式 `False` 关闭;`_indexer_sparse_logits_supported` 放宽为 SM90 直接支持、SM100 保留 DeepGEMM/DeepSelect 检查。
5. CuTe kernel 注册 JIT warmup(`CandidateRowsKernel`、`candidate_paged_logits`、`prepare_candidate_pages`),`enable_jit_warmup` 时预热。

## 关键改动

```python
# vllm/models/deepseek_v41/attention.py(SM90 分派 CandidateIndexer)
+        if self.sparse_logits:
+            if (current_platform.is_cuda()
+                    and current_platform.is_device_capability_family(90)):
+                if not self.candidate_indexer:
+                    return DeepseekV41IndexerBackend
+                from .nvidia.ops.candidate_indexer import CandidateIndexerBackend
+                return CandidateIndexerBackend
```

```python
# vllm/models/deepseek_v41/nvidia/ops/candidate_indexer.py(builder 端校验)
+        if (config.index_n_heads, config.index_head_dim) != (32, 128):
+            raise ValueError(f"{prefix} requires 32 indexer heads of size 128")
+        if self.dcp_world_size != 1 or self.use_pcp:
+            raise ValueError(f"{prefix} does not support context parallelism")
```

## 性能声明

数字均照抄 PR 正文(未独立核实)。环境:8× NVIDIA H20-3e,TP8 with expert parallelism,DeepSeek-V4.1-Flash,PyTorch `2.14.0+cu132`,Triton `3.8.0`,CuTe DSL `4.8.0`;每工作负载 8 请求并发 8、每请求 8,192 实际输出 token、3 次重复;吞吐取 3 run 中位,TPOT/ITL 跨 run 合并采样。

| 指标 | 524,288 输入 | 1,040,384 输入 |
|---|---|---|
| Output throughput | 204.47 → 211.01 tokens/s (**+3.20%**) | 84.70 → 87.43 (**+3.22%**) |
| Mean TPOT | 19.23 → 18.28 ms (−4.97%) | 43.67 → 41.65 (−4.63%) |
| TPOT P95 | 34.16 → 33.05 (−3.23%) | 81.77 → 79.21 (−3.13%) |
| ITL P50 | 27.28 → 22.24 ms (**−18.49%**) | 35.40 → 25.34 (**−28.40%**) |
| ITL P95 | 1,055.38 → 1,042.63 (−1.21%) | 1,690.23 → 1,651.74 (−2.28%) |

吞吐重复区间:512K 为 204.43–204.47 → 210.79–211.10;~1M 为 84.67–84.71 → 87.42–87.45。ITL 度量流式 SSE chunk 间隔(DSpark 一 chunk 可出多 token),speculative acceptance 约每步 6 token。

## 验证与回归风险

- 正确性:96 个测量请求全部完成、48 对生成文本完全一致;GSM8K 128 题 5-shot:126/128 → 127/128(子集回归检查,非精度提升证据);indexer 回归套件 74 passed, 23 skipped;配置检查 9 passed(自动选择、显式覆盖、不支持配置);真权重生成对比通过(TP4+EP、长短 prompt、DSpark CUDA graphs、logprob 对比、非零 acceptance)。
- 新 kernel 测试覆盖 int32 页号全范围、strided cache 布局、CUDA graph replay(重放时新长度/迁移页)、2^33 字节页 stride、与 dense 选择器的 top-k 分数精确一致(并列按分数比较)。
- 回归风险:SM90 路径要求 32 个 128 维 indexer head、FP8 indexer cache、无 context parallelism、无 microbatching,不满足在初始化即 raise(默认 None 也可能 raise);candidate-only prefill 有 32K 压缩阈值,短 prefill 保持 dense;PR checklist 的 Design Fit / Testing 等项在 pack 中显示未勾选。

## 关联

构建于 #62463(Triton decode 打分,yeshihai,Co-authored-by);不同实现路线:#62000(先 gather 再打分);issue #61830;ROCm 侧对应 aiter paged MXFP4 路径(pack 内配置文档提及)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
