# PR #36022: [Kernel] Add FlashInfer MoE A2A Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @leo-cf-tian |
| 合并 | 2026-03-16 |
| 分类 | MoE |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, ready, ci/build, v1, multi-modality, qwen, gpt-oss, nvidia |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/36022 |

## 一句话结论

接入 FlashInfer/TRT-LLM 单侧(one-sided)MoE all-to-all 内核,512 并发下吞吐最高 +14%。

## 动机

EP 部署的 MoE all-to-all 此前可选的 FlashInfer 后端 `flashinfer_all2allv`(two-sided,来自 PR #21003)在大 batch/高并发下不如新的 TRT-LLM gen 内核。flashinfer 的 MoE-A2A API(flashinfer PR #2102)新增了 one-sided all-to-all 内核,针对大 batch 有更好表现;PR #32217 曾做过移植,#32564 的 modular kernel 重构后需要重新接入 top-of-tree。

## 优化点

新增 `--all2all-backend flashinfer_nvlink_one_sided` 后端(仅 nvfp4,要求 Blackwell + flashinfer `trtllm_moe_alltoall` 模块):`FlashInferNVLinkOneSidedManager` 用 flashinfer `Mapping` + `MnnvlConfig`(以 DP group 的 `CustomCommunicator` 为 comm backend)构造 `MoeAlltoAll`,按 per-rank workspace 大小公式(hidden/2 nvfp4 payload + hidden/16 fp8 scale + topk*4 ids + topk*4 weights,combine 为 bf16 hidden*2)一次性初始化。配套 modular kernel `FlashInferNVLinkOneSidedPrepareAndFinalize`:prepare 阶段把 a1q、scale、topk_ids、topk_weights 打包成 payloads 经 `moe_alltoall.dispatch` 单侧发送,fp4 scale 的 swizzle 延迟到通信之后(仅 CUTLASS 路径做 interleave);finalize 阶段按 (ep_size, runtime_max_tokens, hidden) 视图 `combine` 回写输出。同时把 `flashinfer_all2allv` 更名为 `flashinfer_nvlink_two_sided`(旧名作临时别名并告警),CustomCommunicator 补上此前 NotImplementedError 的 `bcast`/`barrier`;7 类 Expert 后端(cutlass/deep_gemm/marlin/triton/rocm_aiter 等)的 `_supports_parallel_config` 不兼容矩阵同步加入 one-sided;`mk_objects.py` 注册该后端与此前缺失的 `TrtLlmNvFp4ExpertsModular` 以获得多 GPU 组合测试覆盖。

## 关键改动

```text
vllm/distributed/device_communicators/all2all.py: FlashInferNVLinkOneSidedManager(MoeAlltoAll workspace 初始化/dispatch/combine)
vllm/model_executor/layers/fused_moe/flashinfer_nvlink_one_sided_prepare_finalize.py: prepare/finalize modular kernel(新文件)
vllm/config/parallel.py + cuda_communicator.py: 后端名注册、two_sided 更名与 one_sided 接线
tests/kernels/moe/modular_kernel_tools/mk_objects.py: one-sided 后端与 TrtLlmNvFp4ExpertsModular 注册
vllm/distributed/device_communicators/mnnvl_compat.py: CustomCommunicator 补 bcast/barrier
```

关键片段(payload 打包与 dispatch):

```python
payloads = [a1q]
if a1q_scale is not None:
    payloads.append(a1q_scale)
payloads.append(topk_ids)
payloads.append(topk_weights)
recv_payloads = self.all2all_manager.moe_alltoall.dispatch(
    token_selected_experts=topk_ids,
    input_payloads=payloads,
    runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
)
```

## 性能声明

nvidia/DeepSeek-R1-NVFP4,DP8 + TP1 + expert parallel,fp8 KV cache,`--all2all-backend` 对比 None / flashinfer_all2allv / flashinfer_moe_a2a,高并发下吞吐有明显提升,512 并发时最高 +14%(具体各并发点数值以 PR 内截图为准,正文未列文字数据)。硬件(正文注明需 Blackwell;测试环境要求 2x Blackwell GPU 与 FlashInfer trtllm_moe_alltoall 支持)与具体 workload 参数(数据集、prompt 数)PR 未在文字部分注明。

## 验证与回归风险

测试:`test_supports_parallel_config_flashinfer_moe_a2a`(CPU only,7 个参数化用例验证新后端的兼容矩阵)、`test_supports_parallel_config_parity_with_all2allv`(与 two-sided 的兼容矩阵一致性,防止单边更新漂移)、`test_modular_kernel_combinations_multigpu`(基于 mk_objects 注册自动生成的多 GPU 组合测试,需 2x Blackwell);正确性可复现:lm_eval GSM8K(local-chat-completions,max_gen_toks 16384,num_concurrent 64)。回归风险:仅 nvfp4 + Blackwell + MNNVL/NVLink 环境可用;`flashinfer_all2allv` 旧名进入弃用路径;one-sided 的 workspace 按 max_num_batched_tokens 预留,超限行为依赖 flashinfer 内核;七类 Expert 后端的不兼容矩阵同时改动,遗漏一处即 IMA(其中 deep_gemm 组合已有前例);future 项:待 A2A manager 支持独立初始化后补充更广 shape 覆盖的多 GPU 专项测试。

## 关联

前序:PR #32217(原始移植)、#32564(modular kernel 重构)、#21003(旧 two-sided all2allv 内核);上游:flashinfer-ai/flashinfer#2102(MoE-A2A one-sided API)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
