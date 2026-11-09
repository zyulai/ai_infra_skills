# PR #59973: [MRV2][Spec Decode] Quantized draft lm_head for drafters that share the target head

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stecasta |
| 合并 | 2026-10-15 |
| 分类 | Speculative Decoding |
| Label | documentation, speculative-decoding, ready, qwen, quantization, mrv2 |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/59973 |

## 一句话结论

给共享 target lm_head 的 drafter 一份私有量化副本(NVFP4/FP8),GB10 上 c=1 decode 提速约 18%,输出不变、仅接受率可微动。

## 动机

MTP/EAGLE drafter 若共享 target 的 lm_head,每个 draft token 都要读整份 head;在 248k x 2560 的 BF16 head 上这是 draft step 中最大的 kernel,带宽受限 GPU 上尤为昂贵。现有量化框架(#51285)只覆盖 linear/MoE 层,不覆盖 `ParallelLMHead`;此前的 draft head 量化尝试(#56577、#47584)是单模型/单方法 + env var 的特例。

## 优化点

新增 opt-in 配置键 `draft_lm_head_quantization: "fp8" | "nvfp4"`(SpeculativeConfig,MRV2 only):drafter 获得共享 target head 的量化私有副本,target 仍用自己的 BF16 head 验证,因此输出不变、只有 acceptance length 可动。实现在新增的 `vllm/model_executor/layers/quantization/online/lm_head.py`:`quantized_lm_head_copy()` 直接从本 rank 的 head shard 量化构造 `ParallelLMHead`(不产生 bf16 副本,target head 不变);"fp8" 复用现有 `Fp8PtpcOnlineLinearMethod`(per-row weight + per-token activation,GB10 上走 `CutlassFP8ScaledMMLinearKernel`);"nvfp4" 新增小型 weight-only 方法 `_LMHeadNvfp4Method`,经 `init_nvfp4_linear_kernel(use_a16=True)` 选 W4A16 kernel(SM12x 上是 Marlin;activation 保持 BF16/FP16)。`ParallelLMHead` 增加可选 `quant_method` 参数。hook 在 `load_eagle_model`(EAGLE/EAGLE3/MTP,含 MTP 每层 `shared_head.head`)。副本是派生态:`refresh_quantized_lm_heads` 在 target 权重被替换时(sleep mode、`reload_weights`、weight transfer)原位重推导,CUDA graph 指针保持有效;draft-only 权重更新期间临时换回未量化 head(share/restore)。probabilistic drafting 保持精确:量化 logits 既被采样也用作 q。支持 CUDA graph 与 TP(每 rank 量化自己的 shard)。

## 关键改动

```text
vllm/model_executor/layers/quantization/online/lm_head.py: 新增 204 行(quantized_lm_head_copy/refresh/share/restore + _LMHeadNvfp4Method)
vllm/config/speculative.py: draft_lm_head_quantization 键 + 校验(仅 eagle/eagle3/mtp,head_dtype 须为 model dtype)
vllm/v1/worker/gpu/spec_decode/eagle/utils.py: load_eagle_model 给 drafter 量化副本
vllm/v1/worker/gpu_worker.py: weight update 全程 share/restore/refresh
vllm/model_executor/layers/vocab_parallel_embedding.py: ParallelLMHead 增加 quant_method 参数
```

## 性能声明

GB10(DGX Spark)x3 节点,SPEED-Bench(Qwen3.8-Flash-Next-NVFP4 + MTP k=3,nightly 0cbac6cd1 + #59753 skinny-GEMM plans):c=1 decode 39.4 → 46.5 tok/s(+17.5% 至 +18.4%),c=4 +9.0% 至 +10.9%;FP8 +10.8% / +6.5%(单节点);平均 AL 2.670 → 2.658(-0.4%,run-to-run noise 内)。更早的 ABBA 测量(#59753 之前):NVFP4 +27.1% c=1、+11.7% c=4,FP8 +21.0% / +9.7%。单 kernel(draft head GEMV 248320x2560,CUDA graph,cold L2):GB10 上 BF16 cuBLAS 5.50-5.58 ms → NVFP4 W4A16 Marlin 1.60-1.64 ms;GB200 上 BF16 191-196 µs → NVFP4 133-149 µs、FP8 W8A8 111-113 µs(FP8 比 Marlin 快 1.3x,GB10 上持平)。额外显存约 0.35 GiB(NVFP4)/ 0.6 GiB(FP8)。独立复现(TP2,2x DGX Spark,MTP k=4):+18.5% (fp8) / +19.5% (nvfp4),greedy 输出逐字节一致。增益随并发缩小(more tokens share each head read)。

## 验证与回归风险

新增 `tests/v1/worker/test_gpu_draft_lm_head.py`(经 LogitsProcessor 对比 BF16 head 的相对误差与 top1 一致率、target 权重更新后原位重推导且指针稳定、layerwise reload 不重量化)与 MTP e2e 正确性测试(输出 80% 匹配、AL >= 0.9x 参考;sleep/reload 后输出与 AL 保持)。风险:副本仅对 MRV2、共享未量化 fp16/bf16 head 且 head_dtype == model dtype 的 drafter 有效,Gemma-4 MTP 被排除;ROCm 无 CUDA W4A16 kernel,只有 "fp8" 可用;vllm-ascend 覆写 `ParallelLMHead.__init__` 缺新参数,Ascend 上需更新 override 否则 soft-fail interface check;greedy 在该模型上不逐位可复现(近局翻转),分类级 AL 可移动约 3%。

## 关联

#56577、#47584(前序特例实现,本 PR 为其通用版)、#59753(SM121 skinny-GEMM plans,当前数字依赖它)、#59740(reduced draft vocabulary,可组合)、#41000/#42124/#58343(checkpoint 内量化 head)、#51285(online quantization framework);同批索引中 #55170 同作者,涉及 NVFP4 linear kernel 选择。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
