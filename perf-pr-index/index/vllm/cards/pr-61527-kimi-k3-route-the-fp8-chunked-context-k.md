# PR #61527: [Kimi-K3] Route the fp8 chunked-context K/V pack through FlashInfer where it has a fused kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @brandonfzhang |
| 合并 | 2026-10-26 |
| 分类 | Quantization \| tier: major |
| Label | ready, nvidia, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/61527 |

## 一句话结论

GB300/SM10x 上 Kimi-K3 的 fp8 KV pack 改走 FlashInfer 融合核，12 heads kernel 快 1.11x–1.29x。

## 动机

FlashInfer 新增公共 API `concat_mla_kv_quant_fp8`，提供与 vLLM 私有 `fused_kimi_k3_mla_kv_concat_quant_fp8` 字节一致的融合 bf16→FP8 MLA context K/V pack，并按 head 数路由（12 local heads 用手写核，其余用 Cake 生成程序）。本 PR 是 #56881 的跟进，让 vLLM、SGLang 等共享一条 kernel 路径，避免每个框架各维护一份拷贝。该 API 在 FlashInfer 侧按 head 数路由：12 local heads（Kimi-K3 的 96 heads 在 TP8 下）走手写 kernel，其余 head 数走生成的 Cake 程序；正文说明它已在 FlashInfer `main`（截至 2026-10-12），但 vLLM 尚未引入含它的正式 release。

## 优化点

在 `vllm/utils/flashinfer.py` 增加带缓存的特性探测 `has_flashinfer_concat_mla_kv_quant_fp8()`。新增 `fused_mla_kv_pack_quant_fp8`：当 FlashInfer 暴露该 API、输入为 bf16、`kv_nope` 连续且设备为 compute capability 10.x 时调用 FlashInfer 的融合实现；否则回退到 vLLM 的 `fused_mla_kv_concat_quant_fp8`（fp16 输入、fp8 k_pe、pre-Blackwell GPU 或无该 API 的 FlashInfer）。`mla.py` 的两条 chunked-context 路径（layer 自身循环与 #56881 的 DCP hook）在 fp8-query 场景改调新函数，bf16-query 路径与其余逻辑不变。无新增环境变量，可用 FlashInfer 的 `FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1` 做 A/B。

## 关键改动

```text
vllm/utils/flashinfer.py: 新增缓存特性探测 has_flashinfer_concat_mla_kv_quant_fp8()。
vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py: 新增 fused_mla_kv_pack_quant_fp8 路由函数。
vllm/models/kimi_k3/nvidia/mla.py: 两处 fp8 chunked-context 路径改调新函数。
tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py: 新增路由与字节一致性测试。
```

```python
def fused_mla_kv_pack_quant_fp8(kv_nope, k_pe, *, nope_dim=128):
    if (kv_nope.dtype == torch.bfloat16
        and k_pe.dtype == torch.bfloat16
        and kv_nope.is_contiguous()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer_concat_mla_kv_quant_fp8()):
        from flashinfer import concat_mla_kv_quant_fp8
        return concat_mla_kv_quant_fp8(
            kv_nope, k_pe.reshape(k_pe.shape[0], k_pe.shape[-1]), nope_dim=nope_dim)
    k_nope, v = kv_nope.split((nope_dim, kv_nope.shape[-1] - nope_dim), dim=-1)
    return fused_mla_kv_concat_quant_fp8(k_nope, k_pe, v)
```

## 性能声明

Kernel 级、GB300、Kimi-K3 部署容器（cold L2、100 次取中位数、沿用 #56881 的 head-by-token 网格），FlashInfer API 对比 vLLM `fused_kimi_k3_mla_kv_concat_quant_fp8`：12 local heads（TP8）、chunk tokens 16–131072 时 1.11x–1.29x faster（手写核）；1–128 heads（非 12）、1536–131072 tokens 时 1.06x–1.26x faster（Cake 程序，flashinfer-ai/flashinfer#6163 表格，B200 与 B300）。12 heads、66k-token serving chunk：120 us → 101 us，逐行字节一致（GB300 run 校验 55 行，另有 FlashInfer 65536-pattern cast test）。端到端约占 93 ms engine step 的 1 ms（70 并发 session），在该路由的 run-to-run 噪声内。

## 验证与回归风险

Test Plan：`pytest tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`、`pytest tests/models/kimi_k3/test_mla_prefill_context.py`。新增 `test_context_kv_pack_routes_to_flashinfer_only_where_it_serves`（stub FlashInfer，bf16/受支持设备走 FlashInfer 且字节一致，fp8 k_pe 或未支持设备不触碰 FlashInfer）与 `test_context_kv_pack_flashinfer_route_is_byte_exact`（真实 FlashInfer、SM10x，3/12/24 heads、1/257/2048 tokens 逐字节一致，其余跳过）。Test Result：H100（vLLM 0.31.0 wheel 叠加本 PR 四个文件，torch 2.13 + CUDA 13.0）46 passed、9 skipped（真实 FlashInfer 行因该 FlashInfer 早于该 API 且 GPU 为 SM90）；`ruff check` / `ruff format --check` 通过。范围：vLLM 引入含该 API 的 FlashInfer 前探针恒为 false、运行时不改行为；Hopper 及更早 GPU 始终使用为 SM90+ 构建的 vLLM kernel。

## 关联

#56881（前序，DCP hook）；FlashInfer 侧 flashinfer-ai/flashinfer#5667、#6163、#6373 提供该融合 API。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
