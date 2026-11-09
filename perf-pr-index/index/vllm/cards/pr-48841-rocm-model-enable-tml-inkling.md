# PR #48841: [ROCm] [Model] Enable TML inkling

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tjtanaa |
| 合并 | 2026-07-27 |
| 分类 | Speculative Decoding |
| Label | performance, new-model, rocm, ready, needs-rebase, ci/build, v1, tool-calling, nvidia, rust |
| Diff | diffstat 缺失;pack 明示 diff 因超长截断(24 个文件未含) |
| 链接 | https://github.com/vllm-project/vllm/pull/48841 |

## 一句话结论

ROCm 首次启用 Inkling(TML)模型 MXFP4 权重,KV page 128 较 page 16 吞吐快 1.29x。

## 动机

Inkling 模型此前只有 `vllm/models/inkling/nvidia/` 实现,ROCm 无路径。本 PR 是 ROCm 上的首次 enablement,并选择更高效的 MXFP4 变体——NVFP4 权重目前只能经低效的模拟路径支持,故权重采用 HuggingFace `lightseekorg/Inkling-MXFP4`;BF16 与 NVFP4 权重的支持仍在持续进行中。

## 优化点

新增完整的 `vllm/models/inkling/amd/` ROCm 实现:`InklingAttention` 复用 `FlashAttentionBackend` 的元数据,核心是 FA4 相对偏置 paged attention(`inkling_fa4_rel_attention`,带 gfx950 Gluon decode/extend 分派与 split-KV decode 启发式);配套 `fused_qkvr_prep` 融合 kernel(附 CUDA graph 微基准脚本)、K/V 短卷积状态、MTP 模块(含 `embed_dual_rmsnorm_cat` 融合输入 kernel);MoE 以 AITER MXFP4 布局加载 Quark OCP MXFP4 权重并保留 AITER padding。`__init__` 按平台分派:ROCm 走 amd,否则走 nvidia;共享定义(logits_processor.py、mtp.py)被测试锁定为与 NVIDIA 参考逐字节一致。

## 关键改动

```text
vllm/models/inkling/amd/(model/attention/mtp/moe/ops): ROCm 完整实现(FA4 rel-attention、qkvr_prep、AITER MXFP4 MoE)
vllm/models/inkling/__init__.py: 按平台分派 amd/nvidia 导出
tests/models/inkling/rocm/: CDNA4 conftest + 对齐/加载/rel-attention/MTP 融合测试
benchmarks/kernels/benchmark_inkling_qkvr_prep.py: qkvr_prep kernel 微基准
```

```python
if current_platform.is_rocm():
    from .amd.model import InklingForCausalLM, InklingForConditionalGeneration
else:
    from .nvidia.model import InklingForCausalLM, InklingForConditionalGeneration
```

## 性能声明

为 enablement PR,数字是启用后的模型评测及同 PR 内 KV page size 对比(Page 16 vs 128;`lightseekorg/Inkling-MXFP4`,`VLLM_ROCM_USE_AITER=1`,TP8,max-model-len 65536):

- Serving:Page 128 vs Page 16,输出吞吐 1,158.54→1,494.37 tok/s(+28.99%),请求吞吐 1.131→1.459 req/s,Mean TPOT 27.08→20.87 ms(−22.93%),Mean ITL −23.19%,TTFT 基本不变(−0.84%);PR 总结"Page 128 约 **1.29x faster** on throughput、**1.30x faster** on per-output-token latency,提升主要来自 decode"。
- GSM8K:flexible 95.22%(page16)/95.83%(page128),strict 94.01%/94.92%。
- BFCL Live AST(1,351 calls,concurrency 128):exact calls 78.76%(1,064/1,351),对比 vLLM Blog 目标 78.61%,落在预期区间 76.42–80.80%。
- MTP-1(`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`):draft acceptance 82.92%/82.88%,mean acceptance length 1.83,GSM8K flexible 95.38%/95.53%。

## 验证与回归风险

新增 `tests/models/inkling/rocm/`:conftest 强制 ROCm gfx950(CDNA4);`test_rel_attention.py` 与纯 PyTorch 参考逐点比对(atol=rtol=3e-2,覆盖 chunked prefill、decode、GQA、local window、split-KV、Gluon 分派与非连续 packed KV 视图);`test_rocm_mtp_input_fusion.py` 数值等价(bf16 eps 容差);`test_model_alignment.py` 断言 amd/nvidia 共享文件逐字节一致;`test_mxfp4_load.py` 验证 TP 分片与 AITER padding。风险:大体量新模型代码;pack 明示 diff 截断(24 文件未含),改动全貌需看 PR;labels 含 needs-rebase 与 ci/build。

## 关联

与既有 NVIDIA 实现(`vllm/models/inkling/nvidia/`)共享模型定义,本 PR 为 ROCm 平台分派新增;正文未引用其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
