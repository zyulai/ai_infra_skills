# PR #42832: [ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @akii96 |
| 合并 | 2026-06-05 |
| 分类 | Quantization(major) |
| Label | rocm, ready, gpt-oss |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/42832 |

## 一句话结论

为 GPT-OSS decode 图新增 torch.compile 融合模式:RoPE(q,k) → static Q FP8 量化 → KV-cache 更新合并进 fused RoPE+KV 原语,conc16 输出吞吐 +4.39%、Median TPOT −5.82%。

## 动机

GPT-OSS 风格的 decode 图中,RoPE、static Q FP8 量化与 KV-cache 更新三个算子相邻出现在热路径上。此前的通用 RoPE+KV 融合模式不匹配"中间夹了 static Q FP8 quant"的图,导致这些算子各自独立执行,存在多余的中间拆分/拷贝;保持 attention 要求的显式依赖顺序不变的前提下把它们折叠进 `fused_rope_and_unified_kv_cache_update`,可省掉中间张量往返。

## 优化点

在 `rope_kvcache_fusion.py` 新增 `RopeStaticQQuantKVCachePattern`:pattern 匹配 split(q,k,v) → RoPE → `static_scaled_fp8_quant(q)` → `unified_kv_cache_update(k,v)` 的 auto_functionalized 图;replacement 改为先调用 `fused_rope_and_unified_kv_cache_update`(RoPE 与 KV 写入在一个 op 内完成),再对融合后的 Q 输出施加 static FP8 量化,从而保留图中的显式依赖排序。注册顺序放在通用 RoPE+KV pattern 之前,并以 `_supports_static_q_fp8_quant_fusion()`(static_scaled_fp8_quant op 存在且平台有 FP8 dtype)做能力门控。另为 ROCm AITER triton rotary op 在 `MatcherRotaryEmbedding.forward_custom` 传 `offsets=None`,使 matcher tracing 的调用形状与 lowered op 一致,保证 AITER 路径也能命中。

## 关键改动

```text
vllm/compilation/passes/fusion/rope_kvcache_fusion.py: RopeStaticQQuantKVCachePattern(layer_name 两种编码路径)注册于通用 pattern 前
tests/compile/passes/test_rope_kvcache_fusion.py: 新增 test_rope_static_qquant_kvcache_fusion 及默认 range 不融合的负例检查
```

replacement 核心节选:

```python
rope_kv_results = auto_functionalized(
    self.FUSED_ROPE_KV_OP, query=q_view, key=k_view, value=v_view,
    positions=positions, cos_sin_cache=cos_sin_cache,
    is_neox=self.is_neox, layer_name=layer_name)
q_after = rope_kv_results[1]
q_fp8 = auto_functionalized(torch.ops._C.static_scaled_fp8_quant.default,
    result=q_fp8, input=q_after_flat, scale=q_scale, group_shape=(-1, -1))
```

## 性能声明

模型 `openai/gpt-oss-120b`,ROCm(GPU 型号 PR 未注明):

- 输出吞吐:conc 16/32/64 分别 +4.39% / +1.41% / +1.75%
- Median TPOT:conc 16/32/64 分别 −5.82% / −1.74% / −0.62%
- GSM8K LM-eval spot-check(`--limit 256`,chat-template 路径):baseline 与 PR flexible exact_match 均为 0.9062 ± 0.0183

其余 workload 细节(input/output 长度等)PR 未注明。

## 验证与回归风险

新增 `test_rope_static_qquant_kvcache_fusion`(ROCm AITER UNIFIED attn 后端、aiter triton rope 开/关、kv_cache_dtype auto/fp8 参数化):对比融合前后 q/k/v/kv_cache 数值,断言 `fusion_pass.matched_count == 1`、融合后仍含 static quant op,并以 monkeypatch 关掉 static-Q 支持做负例对照(通用 pattern 匹配数为 0)。FX-dump 确认 decode 路径出现 `fused_rope_and_unified_kv_cache_update`。风险面:仅影响开启 `fuse_rope_kvcache` 且 token range ≤ `max_token_num` 的编译段;新 pattern 先于通用 pattern 注册,若 capability guard 误判可能遮蔽旧匹配路径;依赖 `is_applicable_for_range` 对大 token 范围保持不融合。

## 关联

同区域:既有 `RopeReshapeKVCachePattern`/`fused_rope_and_unified_kv_cache_update` 融合体系;依赖 #57097 等融合原语系列(具体编号素材未列出)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
