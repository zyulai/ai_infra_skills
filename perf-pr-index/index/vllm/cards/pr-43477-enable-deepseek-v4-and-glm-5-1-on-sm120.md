# PR #43477: Enable DeepSeek V4 and GLM-5.1 on SM120

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lucifer1004 |
| 合并 | 2026-06-22 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, speculative-decoding, ready, ci/build, v1, deepseek, nvidia, verified |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 因超长截断（25 个文件未展开），可见部分含 `docs/design/attention_backends.md`、`vllm/model_executor/layers/attention/mla_attention.py`、`vllm/model_executor/layers/fused_moe/deep_gemm_utils.py`、`fused_moe/experts/{deep_gemm_moe,batched_deep_gemm_moe}.py`、`fused_moe/oracle/mxfp4.py`、`quantization/utils/fp8_utils.py`、`warmup/deep_gemm_warmup.py` 及测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/43477 |

## 一句话结论

在消费级 Blackwell（SM120，RTX PRO 6000 等）上打通 DeepSeek V4 与 GLM-5.1：FlashInfer SM120 sparse MLA 后端 + DeepGEMM MXFP4 MoE + autotune warmup，DSv4-Flash TP2 MTP=2 最高 633.3 tok/s。

## 动机

DeepSeek V4（sparse MLA + sparse indexer）与 GLM-5.1 的推理路径此前只覆盖数据中心 GPU（SM90/SM100），消费级 Blackwell SM120 缺少三块拼图：sparse MLA 的 FlashInfer 后端、DeepGEMM 的 MXFP4 grouped-GEMM 支持、以及配套的 kernel warmup/autotune。不解决则新一代 MoE+MLA 模型无法在消费卡上以接近数据中心栈的内核效率运行。

## 优化点

五条线：其一，将 DSv4 sparse MLA 路由到新的 FlashInfer SM120 sparse MLA wrapper/backend，注册 `FLASHINFER_MLA_SPARSE_SM120`（bf16，KV `auto/fp8/fp8_e4m3/fp8_ds_mla`，block 64/256），并把 `FLASHINFER_MLA_SPARSE_DSV4` 扩展到 SM12x（NVIDIA 上 SM12x 默认 DSV4 后端从 `FLASHMLA_SPARSE_DSV4` 改为 `FLASHINFER_MLA_SPARSE_DSV4`）；KV dtype 经 `_canonicalize_sparse_mla_kv_cache_dtype` 统一为 `fp8_ds_mla`。其二，DeepGEMM MXFP4 MoE 支持 SM120（`DeepGemmFP4Experts`、packed UE8M0 activation scales 扩到 capability family 120），MXFP4 scale 经 `deepgemm_post_process_weight_scale_block` 按目标架构打包布局。其三，grouped-GEMM 对齐启发式重构：`compute_aligned_M_and_alignment` 在 expert tokens 不在 CPU 时用 `get_theoretical_mk_alignment_for_contiguous_layout` 收缩 per-call alignment 并把 padding 上限从 `local_num_experts*(align-1)` 收紧为 `min(M*num_topk, local_num_experts)`，GEMM 调用包在 `mk_alignment_scope(align_used)` 中——防止 cudagraph replay 下 DeepGEMM 的 BLOCK_M 启发式从 m_indices 选错 expert id。其四，接入 DSv4 专属 kernel warmup，含 warmup 期间的 FlashInfer decode autotune。其五，CMake 为本地/拉取的 DeepGEMM 与 QuTLASS 源码树接线。依赖上游：FlashInfer#3395 与 DeepGEMM#324。

## 关键改动

```text
vllm/model_executor/layers/attention/mla_attention.py: SM120 sparse MLA 注册与 KV dtype 规范化（fp8_ds_mla）
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py: compute_aligned_M_and_alignment + ep_scatter 的 ALIGN_M 参数化
vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py: DeepGEMM FP8/FP4 MoE 支持 SM120；mk_alignment_scope 包 GEMM
vllm/model_executor/layers/quantization/utils/fp8_utils.py: deepgemm_post_process_weight_scale_block 统一 scale 布局转换
docs/design/attention_backends.md: 新 backend 表项与 DSV4 默认后端说明
```

```python
max_active_experts = min(M * num_topk, local_num_experts)
M_sum = (M * num_topk) + max_active_experts * (alignment - 1)
M_sum = round_up(M_sum, alignment)
return M_sum, alignment
```

## 性能声明

性能表为 SM120 bring-up 数据（非前后对比）。硬件 NVIDIA RTX PRO 6000 Blackwell Server Edition（96GB，SM120），random dataset，ISL=8000/OSL=1000。DeepSeek-V4-Flash TP=2：无 MTP 并发 1 输出吞吐 88.1 tok/s（TPOT 10.65ms）、并发 32 为 398.6 tok/s；MTP=1 并发 32 为 566.3 tok/s（acceptance length 1.97）；MTP=2 并发 1 为 158.5 tok/s、并发 32 为 633.3 tok/s（TPOT 44.76ms，acceptance length 2.71）。GLM-5.1-NVFP4 TP=8：并发 1 为 43.7 tok/s、并发 32 为 281.1 tok/s。精度：GSM8K Pass@1 95.0%；GPQA-Diamond Pass@1 无思考 73.2%、最大思考 87.4%（384K 上下文，2 题因输出长度上限失败）。

## 验证与回归风险

测试：新增 `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py`（backend 名称/枚举映射、GLM block size 校验通过）；`test_sparse_mla_backends.py` 改用 `FlashInferMLASparseTRTLLMBackend` 并按 `supports_compute_capability` 跳过非 SM10.x；删除 `test_flashinfer_autotune_cache.py`；文档生成脚本支持非连续 capability 列表（如 "10.x, 12.x"）。风险面：影响 SM100 上 DeepGEMM MoE 的 workspace 对齐与 GEMM scope 逻辑（所有 Blackwell 共用代码路径）；SM12x 上 DSV4 默认 sparse MLA 后端从 FlashMLA 切换为 FlashInfer；上游依赖未合入前需 `DEEPGEMM_SRC_DIR`/分支版 FlashInfer 本地构建。

## 关联

上游依赖：flashinfer-ai/flashinfer#3395、deepseek-ai/DeepGEMM#324。索引内同区域（DSV4/sparse MLA）：#51538（DSV4 sparse MLA 端到端修复）、#52861（DSA 模型路由到 CUDA 后端）。PR 正文未列其他 vLLM 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
