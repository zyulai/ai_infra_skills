# PR #34597: [Kernel] Add FP8 KV cache support to Triton MLA decode attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @grimulkan |
| 合并 | 2026-03-12 |
| 分类 | Quantization（tier: major） |
| Label | documentation, ready, v1 |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 4 个文件（triton_mla.py、triton_decode_attention.py、测试、attention_backends.md 文档，约 +180/-8 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/34597 |

## 一句话结论

给 Triton MLA decode attention 加 FP8 KV cache 支持（sm120 上唯一的 MLA 后端此前直接 NotImplementedError），省一半 KV 显存、换约 2 倍并发容量，计算仍走 bf16。

## 动机

Triton MLA 后端是 sm120（Blackwell 消费级 GPU）上唯一可用的 MLA 后端，但此前对 fp8/fp8_e4m3 KV cache 抛 `NotImplementedError`——`forward_mqa` 与 backend 构造两处都显式拒绝。MLA 模型（如 Kimi K2.5）在消费级 Blackwell 上因此无法用 FP8 KV cache 压缩 KV 显存。

## 优化点

采取"dequant-on-load"方案：kernel 加载 KV tile 后立即反量化为 bf16 再做点积，是省显存但不省算力的 Mode 1。(1) `TritonMLABackend.supported_kv_cache_dtypes` 增加 fp8/fp8_e4m3，backend 构造时对量化 KV 设 `supports_quant_query_input = False`（保持 BF16 query + FP8 KV 组合）；(2) `k_scale`/`v_scale` 从 layer 穿透到两个 stage1 Triton kernel（MHA 非分组路径 `_fwd_kernel_stage1` 与 GQA/MLA 分组路径 `_fwd_grouped_kernel_stage1`，后者含 kpe 分支），kernel 内以 `k.dtype.is_fp8()` 运行时分支做 `(k.to(tl.float32) * ks).to(q.dtype)`；(3) `decode_attention_fwd` 入口在 scale 为 None 时默认 1.0 张量，BF16 cache 下 `is_fp8()` 分支为 no-op，调用侧"永远传 layer scales"；(4) 文档表更新 TRITON_MLA 支持的 KV dtype。已知取舍：不是真正的 fp8 tensor core op，只省显存不提算力；DCP + FP8 不支持（MLA common layer 既有限制，#34795 修复）。

## 关键改动

```text
vllm/v1/attention/ops/triton_decode_attention.py: 两个 stage1 kernel 增加 k_scale/v_scale 参数与 is_fp8() 反量化分支
vllm/v1/attention/backends/mla/triton_mla.py: 解除 FP8 KV 拒绝，supports_quant_query_input=False，forward_mqa 传 layer._k_scale/_v_scale
tests/kernels/attention/test_triton_decode_attention.py: 新增 16 个 FP8 参数化用例（per-tensor scale 量化对拍 BF16 参考）
docs/design/attention_backends.md: TRITON_MLA KV dtype 表更新
```

kernel 内反量化：

```python
if k.dtype.is_fp8():
    k = (k.to(tl.float32) * ks).to(q.dtype)
qk = tl.sum(q[None, :] * k, 1)
```

## 性能声明

无 kernel 级速度数字。作者声明：归一化生成速度（不计 fp8 KV 带来的潜在 **2x higher concurrency**）与 bf16 基本相同——该方案的预期行为是省显存不省算力。精度：GSM8K 5-shot（Kimi K2.5，16×RTX 6000 Pro，DCP=1，lm_eval local-completions 500 并发）：`--kv-cache-dtype auto`（BF16）0.9378±0.0067 → `fp8` 0.9393±0.0066（约 0.15 分差异，在预期容差内）。

## 验证与回归风险

`pytest tests/kernels/attention/test_triton_decode_attention.py`：112 全过（96 既有 BF16 + 16 新 FP8）。FP8 用例以 per-tensor scale（amax/448，clamp −448/448）量化 K/V 后与 BF16 参考对拍，容差 `atol=5e-1, rtol=1e-2`（与 test_mla_backends.py 的 backend correctness 一致）。风险面：(1) per-tensor scale（非 per-block/per-head），对异常分布的 KV 精度敏感，容差较宽；(2) 反量化在每次 tile load 后做，fp32 乘加转回 bf16 的开销使 kernel 略慢于纯 bf16 路径（作者称归一化速度"基本相同"为定性说法）；(3) DCP + FP8 不支持，部署时需避开；(4) kpe（rope 尾部）分支同样走 ks，两层路径的 scale 语义需一致。

## 关联

相关：#34795（DCP+FP8 修复，本卡 Known limitations 引用）。同分类 FP8 KV 相关：#60895（FP8 MegaMoE）、#14152（Triton 后端含 fp8 KV path）。TRITON_MLA 在 sm120 的地位使其成为 Blackwell 消费级 MLA 部署的关键路径；素材未列其他直接关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
