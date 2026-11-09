# PR #26852: [AMD]Reuse fused FP8 KV cache write on standard aiter prefill/decode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @IzacharyI |
| 合并 | 2026-07-16 |
| 分类 | Quantization |
| Label | amd, run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/26852 |

## 一句话结论

把融合 bf16→FP8 KV cache 写入 kernel 从 unified attention 路径复用到标准 aiter prefill/decode,prefill KV 写 kernel 最高提速 6.40x。

## 动机

融合 KV 写 kernel `launch_reshape_and_cache_flash` 之前只在 unified-attention 路径(`use_triton_unified_attention`)启用;标准 aiter 路径(经 `paged_attention_ragged` 读取)的 FP8 KV 写回退到 `set_kv_buffer`,即 separate divide + cast + store_kvcache,每个 attention 层的 K/V 都有额外 elementwise kernel 与中间 DRAM 流量。KV 写只产出 fp8 cache、与 attention 读取路径无关,因此融合 kernel 可以直接复用到标准路径;同时修复一个潜在的 scale 正确性缺口。

## 优化点

在 `aiter_backend.py` 引入共享判定 `_use_fused_fp8_kv_write`(FP8、非 MLA、非 SWA、K/V head 数与 head_dim 一致)。Decode 侧把融合写的 gate 从 `use_triton_unified_attention and kv fp8` 放宽为 `kv fp8 and not use_mla`,标准路径也走 `launch_reshape_and_cache_flash`;Prefill 侧为非 SWA、非 MLA 的 FP8 情形新增同一融合写(此前用 naive `set_kv_buffer`)。正确性上向 `launch_reshape_and_cache_flash` 传入 `k_scale/v_scale`:既有 decode FP8 分支(#23620)此前漏传,仅在 KV descale 为 1.0 时碰巧正确。另修正 `v_descale` 回退取 `self.k_scale` 的笔误为 `self.v_scale`。MLA 明确排除,保留自己的 KV 写路径(set_kv_buffer 无 scale 参数)。

## 关键改动

```text
python/sglang/srt/layers/attention/aiter_backend.py: _use_fused_fp8_kv_write 判定与 prefill/decode 融合写分支
test/registered/attention/test_fused_fp8_kv_write.py: 新增 kernel 精度与分发测试(437 行)
```

```python
def _use_fused_fp8_kv_write(self, layer: RadixAttention) -> bool:
    return (
        self.kv_cache_dtype == fp8_dtype
        and not self.use_mla
        and not self.use_sliding_window_kv_pool
        and layer.tp_k_head_num == layer.tp_v_head_num
        and layer.qk_head_dim == layer.v_head_dim
    )
```

## 性能声明

kernel 速度(fused vs naive):decode bs=1 nkv=1 39.75→37.51 us(1.06x),bs=16 1.05x,bs=32 nkv=1 1.04x,bs=32 nkv=4 1.06x;prefill q=8000 nkv=1 100.41→41.32 us(2.43x),nkv=4 271.33→42.38 us(6.40x)。kernel 精度:与 fp32-divide 参考逐位一致(mismatch 0.0000%),vs naive bf16 写 decode ~1.7%、prefill ~2.5% 位不同,mean abs err fused ≤ naive。GSM8K(Qwen3.5-397B-A17B,TP8,1319 题 5-shot greedy):naive FP8 写 93.1% / 本 PR 融合写 92.9% / bf16 KV 参考 94.2%,差距来自 FP8 KV 量化本身。e2e(TP8, bs=1, out=128):fp8-KV in=8000 median TTFT 400.2 ms、TPOT 8.60 ms;bf16-KV 408.3 ms、8.70 ms。硬件未在 pack 中注明(ROCm 相关)。

## 验证与回归风险

新增 `test_fused_fp8_kv_write.py`:kernel 级对 fp32-divide 参考的一致性、对 naive 无精度回退、只写目标 slot;分发级验证 v_scale 回退到 `self.v_scale`、head_dim 不匹配回退 set_kv_buffer 且透传 descale、MLA 不带 scale 调用。风险面:改写所有标准 aiter FP8 路径的 KV 写入,head-dim 不匹配/SWA/MLA 有显式回退;但 FP8 KV 量化本身带来约 1.3pp GSM8K 差距(与 bf16 参考)。pack 显示 CI Base 与 Extra 均失败。

## 关联

前序 #23620(unified 路径的 decode FP8 融合写,遗漏 scale 传参由本 PR 修复)。同区域(AMD attention/quant)其他 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
