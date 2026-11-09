# PR #24692: feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AliceChenyy |
| 合并 | 2026-06-01 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, deepseek, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用(另有 6 个文件因超长未包含在 pack 中) |
| 链接 | https://github.com/sgl-project/sglang/pull/24692 |

## 一句话结论

为 SM120 桌面 Blackwell(RTX PRO 6000/5090)补齐 DeepSeek-V4/V3 的 Triton fallback 内核并打通 CUDA graph,BS=1 decode 达 11.40 tok/s。

## 动机

SM120 是桌面 Blackwell,缺少服务器级特性:无 TMEM、无 tcgen05、无 DeepGEMM 支持,也没有 NVSwitch。此 PR 之前 SGLang 完全无法在 SM120 上运行 DSv4(DeepGEMM JIT 崩溃、无 MXFP4 MoE 支持)。目标是在工作站 GPU 上开放 DSv4,给开发者/研究者本地推理能力。

## 优化点

为 SM120 补齐关键路径内核并以 `is_sm120_supported()` 守卫(声称对 SM100/SM103 路径零影响):

1. **新内核(7 个文件)**:Triton fused MXFP4 dequant+GEMM MoE 内核(per-(token,expert) slot GEMV,融合 FP4 解包与 GEMV,无 `.unique()/.item()/.nonzero()`,固定 grid,兼容 CUDA graph;受 SM120 99KB SMEM 约束);Triton FlashMLA sparse decode 内核(tiled 向量化,三种 typed view 处理 FP8 nope/UE8M0 scale/BF16 rope 区段,base-2 online softmax,SWA 与 extra cache 分算后 LSE 合并);FP8 paged MQA 向量化 torch 实现(wq-precompute、无 .item(),CUDA graph 兼容);另有三个 PyTorch fallback 与 MQA fallback 单测。
2. **SM120 guards(10 个修改文件)**:DeepGEMM/tilelang MHC 在 SM120 上禁用(sm_version==120 直接返回 False);NSA backend 默认 tilelang、跳过 DeepGEMM metadata 分配;deepseek_v4_backend.py 增加 FlashMLA SM120 adapter(跳过 tile scheduler metadata,改走 `flash_mla_with_kvcache_sm120`);MoE 在 SM120 自动选 marlin backend;修复 3 处 CUDA-graph-breaking 路径(`.unique()/.item()/.nonzero()` 改向量化)。
3. **Bug fix**:FlashMLA Triton 内核处理 KV cache `uint8` dtype(上游从 float8_e4m3fn 改动)。

环境变量:`SGLANG_SM120_TRITON_FLASHMLA=1`(默认开)、`SGLANG_SM120_MQA_FALLBACK=0`(默认关)。

## 关键改动

```text
python/sglang/srt/layers/attention/flash_mla_sm120_triton.py: 新增 tiled Triton
  sparse decode 内核(2D gather、tile 级 online softmax、LSE 合并)
python/sglang/srt/layers/attention/flash_mla_sm120.py: SM120 FlashMLA 入口,
  triton/torch 双后端分发
python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py: 新增
  fused MXFP4 dequant+GEMV/GEMM Triton 内核(per-slot,CUDA graph 兼容)
python/sglang/srt/layers/attention/deepseek_v4_backend.py: SM120 adapter,
  跳过 flashmla metadata,改调 flash_mla_with_kvcache_sm120
python/sglang/srt/layers/deep_gemm_wrapper/configurer.py: SM120 禁用 DeepGEMM
python/sglang/srt/layers/attention/dsv4/indexer.py: 新增
  fp8_paged_mqa_logits_torch_sm120 向量化实现
```

## 性能声明

PR 正文,未独立核实。硬件 8× RTX PRO 6000,TP=8,开 CUDA graph:

| 环境 | GSM8K 5-shot (200q) | Decode BS=1 | TPOT |
| --- | --- | --- | --- |
| sglang:dev-cu13(sgl-kernel 0.4.2.post1,PyTorch 2.11+cu130) | 99.0% | 11.40 tok/s | 87.7ms |
| nightly-dev-20260430(sgl-kernel 0.4.1,PyTorch 2.9.1+cu129) | 98.0% | 10.26 tok/s | 97.5ms |

- CUDA graph speedup **2.4×**(vs 不开:4.36 → 10.26 tok/s,旧镜像)。
- 新镜像比旧镜像 TPOT 快 **11%**。
- 内核级声明:Triton MXFP4 MoE 较 PyTorch per-GEMM 快 **4.1×**;Triton FlashMLA 较 FlashInfer fallback 快 **3.2–5.4×**。
- CUDA graph capture 全部 batch size 成功;PDL 在 SM120 上验证正确(GSM8K 10/10)。

## 验证与回归风险

- 新增 `test_sm120_mqa_fallback.py` 单测;GSM8K 5-shot 200q 两个镜像均为作者实测;CI 无 SM120 runner(Checklist 未勾选),本地 8×RTX PRO 6000 验证。CI 状态(pack 记录):Base 与 Extra 均通过。
- SM120 分支由 `is_sm120_supported()` 守卫,PR 声明对 SM100/SM103 零影响,但 `deepseek_v4_backend.py`/`indexer.py` 的 dispatch 改动仍需其他架构回归。
- MQA torch sm120 实现硬编码 head_dim=128、block_size=64 布局,并要求 `clean_logits == False`。
- 本 PR 是 #24047(面向 deepseek_v4 分支)向 main 的 rebase,内核代码经 diff 验证一致。

## 关联

#24047(前序 PR,deepseek_v4 分支版本)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
