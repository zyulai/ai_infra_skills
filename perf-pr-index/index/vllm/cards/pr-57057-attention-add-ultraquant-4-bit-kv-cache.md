# PR #57057: [Attention] Add UltraQuant 4-bit KV cache backend (FlyDSL D=256)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aditi-amd |
| 合并 | 2026-10-06 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, intel-gpu, ready, quantization, verified |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；pack 中 diff 被截断，注明省略 10 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/57057 |

## 一句话结论

新增 ultraquant_4bit 4-bit KV cache 后端，KV 密度约 2× fp8，高并发长上下文吞吐最高 +275%。

## 动机

长上下文/agentic 服务的瓶颈是 KV cache 容量：并发升高后 cache 被占满、开始驱逐（evicting），吞吐随之坍塌。UltraQuant 把 KV cache 存成 4 bit，相对 fp8 约 2 倍 KV 密度，让更多并发请求在驱逐前装进显存，decode 在高负载下仍保持内存效率。PR 正文给出与 8-bit 基线（KV8）的对照：低并发两者持平，KV8 触顶驱逐后 UQ 持续扩展。

## 优化点

- KV 存储格式：FP4 E2M1 codes + UE8M0（2 的幂）per-group-32 scales，K 存储时做 Hadamard 旋转（V 不旋转），提高量化后的数值质量。
- Decode 走 FlyDSL `D=256` scaled-MFMA kernel（gfx950），覆盖 GQA group sizes {6, 8, 16}；不满足条件（head_size 非 256、有 sinks、sliding window、FlyDSL 未加载）或非 ROCm 平台时回退 Triton unified attention，另有数值参考实现。
- Continuation prefill 分两档：小 chunk 用 Triton unified kernel，大 chunk 先整体 dequant 再走 FlashAttention。
- 实现上通过继承 `TurboQuantAttentionImpl`/`TurboQuantAttentionBackend` 复用 forward 编排、metadata、cache-layout 寻址与 CUDA-graph pre-warm，只覆写 store/decode/prefill 三类 format-specific seams；同时注册为独立的 `AttentionBackendEnum.ULTRAQUANT` 与独立 dtype `ultraquant_4bit`（torch.uint8），TurboQuant 侧不含 UltraQuant 知识。

## 关键改动

```text
vllm/v1/attention/backends/ultraquant_attn.py: UltraQuant 后端与实现，覆写 store/decode/prefill seams
vllm/v1/attention/ops/ultraquant/: FP4/UE8M0 format、Hadamard store、dequant、Triton unified attention、reference
vllm/v1/attention/ops/flydsl_ultraquant_decode.py: FlyDSL D=256 decode kernel/launcher 与 eligibility 路由
vllm/config/cache.py + vllm/utils/torch_utils.py: 注册 ultraquant_4bit KV dtype（uint8）
vllm/platforms/rocm.py + vllm/platforms/interface.py: backend 优先级、layout 共享校验、hybrid 模型 page-size 计算
```

后端 docstring 概括了三个 seam 的分流策略：

```python
  * store         -> ultraquant_store (rotate K, pack FP4 + UE8M0 scales)
  * decode        -> FlyDSL D=256 scaled-MFMA kernel, Triton unified fallback
  * continuation  -> small chunk: Triton unified; large chunk: dequant + FA
```

## 性能声明

Accuracy（GPQA-Diamond，多 seed 平均，KV 格式为唯一变量）：Qwen3.8 UQ 93.27% vs KV8 fp8 92.42%；Qwen3.6-27B UQ 86.87% vs KV8 85.61%，与 fp8 基线持平略优。

Throughput-latency Pareto（262K-token agentic replay，MI355X）：

- Qwen3.8（GQA-16, TP=8）：C=8/16 时 UQ 低 2.7%；C=32 起 +3.0%；C=36 +33.1%；C=38 时 KV8 cache 达 100%（evicting）、吞吐 225.4 tok/s，UQ 414.6 tok/s（+84.0%）；C=42 +275%（430.7 vs 114.7 tok/s）。
- Qwen3.6-27B（GQA-6, TP=2）：C=42 +9.0%（592.0 vs 543.1 tok/s）且 ITL 更低；UQ peak KV 约为 KV8 的一半。

PR 自注：benchmark 跨不同 vLLM/AITER/FlyDSL base 测量，绝对吞吐/延迟（含 8-bit 基线）仅供参考，准确率与 UQ-vs-KV8 相对趋势才是结论。

## 验证与回归风险

- 新增 `tests/kernels/ultraquant/`：config、store 与 reference 逐位一致、dequant、Triton unified decode vs reference、FlyDSL D=256 decode vs reference、seam contract 测试（用 AST/inspect 钉住 TurboQuant seam 签名与 `_tq_*` 属性契约，防止基类改动静默破坏 UltraQuant）。
- 风险路径：需要 FlyDSL-capable ROCm build（gfx950）才能走 D=256 decode kernel；继承 TurboQuant 的 FlashAttention ≥ 3 不兼容（config 层强制覆写为 FA2）；boundary attention layers 保持 native dtype，layout 共享校验影响其他 backend 的选择；FlyDSL 仅支持 head_size=256 + GQA {6,8,16}，其余形状靠 Triton fallback；platforms/interface 的 page-size 分支改动影响 hybrid 模型 KV spec 计算。

## 关联

- 前序：TurboQuant 4-bit KV cache 系列（`turboquant_4bit`/`turboquant_k3v4_nc`/`turboquant_3bit_nc` 等 dtype 与 `TurboQuantAttentionBackend` 已存在；本卡素材未含其 PR 编号）。
- 同区域：本批同类 Attention/Kernel 优化 PR #51739（MLA cache gather）、PR #39616（ROCm AITER MLA + Eagle3）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
