# PR #45739: [Perf] [AMD] Qwen3-Next: fused QK Gemma RMSNorm + partial RoPE on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @karverma-amd |
| 合并 | 2026-10-27 |
| 分类 | Attention / Kernel (major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 为 1 个文件(qwen3_next.py)的接入改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/45739 |

## 一句话结论

在 gfx950 上把 Qwen3-Next 全注意力 prologue 的 Q/K Gemma RMSNorm + partial RoPE + Q||gate 拆分融合进单个 aiter op,TPOT 降 2.3-3.0%。

## 动机

Qwen3-Next 的 full-attention prologue 在 native 路径要执行三个小 kernel:qkv 输出中 per-head 的 `[Q||gate]` 拆分、Q/K 的 Gemma RMSNorm(`x*(1+w)`)、以及 rotary_dim 64 的 partial NeoX RoPE。这些 launch 小、占比高,在 AMD gfx950(MI355X)上是可合并的开销。ROCm/aiter 已提供 `fused_attention_qk_norm_rope` 融合 op,但 SGLang 的 Qwen3-Next 实现没有接上。

## 优化点

在 `qwen3_next.py` 中新增 `forward_prepare_fused`:一次 `qkv_proj` 后直接调用 aiter 的 `fused_prologue`,单次 launch 完成 [Q||gate] 拆分、Q/K Gemma RMSNorm 与 partial RoPE,返回 q/k/v/gate 四个张量;`self_attention` 中仅当 `_use_fused_prologue` 且 `attn_output_gate` 开启时尝试融合路径,失败则落回原 `forward_prepare_native`。门控设计:SGLANG_USE_AITER 环境变量 + `is_hip()` + `is_gfx95_supported()` 三重判断,其他 HIP 架构(如 gfx94x)不走融合路径、也不会为融合多跑一次 qkv_proj;aiter 懒加载且仅在 HIP 到达,导入失败(`ImportError`)时返回 None 走 native,lru_cache 缓存探测结果;shape 级支持用 `prologue_supported()` 探测而非 assert,任何不支持的 dtype/宽度/架构都 fail-closed 回退。attention backend 仍照常写 KV cache。

## 关键改动

```text
python/sglang/srt/models/qwen3_next.py: 新增 _fused_attention_prologue 懒加载探测与 forward_prepare_fused 融合 prologue,self_attention 接入
```

融合路径核心:

```python
def forward_prepare_fused(self, positions, hidden_states):
    ops = _fused_attention_prologue()
    if ops is None:
        return None
    fused_prologue, prologue_supported = ops
    qkv, _ = self.qkv_proj(hidden_states)
    cos_sin_cache = self.rotary_emb.cos_sin_cache
    ok, _ = prologue_supported(qkv, self.q_norm.weight, self.k_norm.weight, positions, cos_sin_cache)
    if not ok:
        return None
    query, gate, key, value = fused_prologue(qkv, self.q_norm.weight, self.k_norm.weight, positions, cos_sin_cache)
    return (query.reshape(-1, self.q_size), key.reshape(-1, self.kv_size),
            value.reshape(-1, self.kv_size), gate.reshape(-1, self.q_size))
```

## 性能声明

测试条件:Qwen3-Next-80B-A3B-Instruct-FP8,MI355X 2×TP4,nightly `v0.5.22-rocm724-mi35x-20261021`,Triton 3.8.0。

- E2E:Median TPOT **−2.3 至 −3.0%**(concurrency 1/8/32),throughput **+2.3 至 +7.9%**;cuda-graph capture 与 sustained concurrency 8/32/64 负载下稳定。
- 准确率:GSM8K 1319 题 5-shot 持平(fused 0.941 / native 0.948,0 invalid)。
- 复现性:同一 nightly 换 stock Triton 3.7、以及 v0.5.21 + Triton 3.8,均复现 −2.5 至 −3.0% TPOT 与 GSM8K 持平。
- 融合臂确认实际走进 fused op,对照臂验证 fail-closed 到 native。

kernel 级单 launch 耗时对比 PR 未注明。

## 验证与回归风险

无新增测试文件;验证为多环境 E2E TPOT/吞吐对比 + GSM8K + engagement/fail-closed 双向确认。CI 状态:PR Test (Base) 通过,PR Test (Extra) 失败,AMD ROCm 10 排队中。风险面:依赖 ROCm/aiter 提供 `fused_attention_qk_norm_rope`(PR 附带 aiter 侧分支链接);stacked on #45738(`smallm_fp8_blockscale_gfx950` import 修复,是 Qwen3-Next FP8 在 gfx950 上服务的前提,本 PR 第一个 commit 即该修复);融合仅在 `attn_output_gate` 开启 + gfx950 + SGLANG_USE_AITER 时生效,NVIDIA 与其他 HIP 架构路径不受影响。

## 关联

前序/依赖:#45738(gfx950 smallm fp8 blockscale import 修复,本 PR 的 stacked-on 基座)。同区域:Qwen3-Next attention/linear-attention kernel 优化、aiter 融合 op 系列(pack 未列具体编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
