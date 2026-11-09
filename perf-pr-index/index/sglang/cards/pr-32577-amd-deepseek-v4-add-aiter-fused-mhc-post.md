# PR #32577: [AMD] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @karverma-amd |
| 合并 | 2026-08-22 |
| 分类 | Attention / Kernel(major) |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 3 个文件(AMD fused mHC 模块、deepseek_v4.py、新增单元测试),2 个文件因超长未收入 |
| 链接 | https://github.com/sgl-project/sglang/pull/32577 |

## 一句话结论

DeepSeek-V4 在 gfx95 上新增 aiter HIP 融合 mHC post+pre kernel 并接入跨层边界 dispatch,MI355X TP4 下 output throughput +0.8-1.8% 且精度保持。

## 动机

DeepSeek-V4 的 mHC(multi-head mixing)在每个 attn/MoE 边界有 `mhc_post` + 下一层 `mhc_pre` 两步。SGLang 已有 TileLang 与 Triton 两条融合路径,但 MI355X(gfx950)上 AMD 自家的 aiter HIP kernel(`aiter.ops.mhc.mhc_fused_post_pre`)尚未接入;且跨层融合的开启条件原来只看 TileLang flag,gfx95 用户拿不到该收益。

## 优化点

在 `deepseek_v4_fused_mhc.py` 中新增 `try_aiter_fused_mhc_post_pre()` 包装 aiter kernel:aiter kernel 自身按 arch/token 数做 fused-vs-unfused 门控,wrapper 不重复 token 阈值逻辑;import/runtime 失败仅 warn 一次并永久禁用融合路径。`try_mhc_fused_post_pre_boundary()` 实现 aiter → Triton 的优先级 dispatch(两个 kernel 对 `hc_fn` 的朝向要求相反:Triton 需转置,aiter 用原生布局,经 `fn_transpose` 参数统一)。`is_cross_layer_mhc_fusion_enabled()` 把开启条件放宽为 "TileLang 路径可用 或 aiter gfx95 路径可用"。模型侧把 attn 侧、FFN 侧及 `op_mhc_post_attn_pre_mlp` 三个边界统一走 `_apply_mhc_post_pre_boundary()`,无融合 kernel 可用时回退 unfused `hc_post`+`hc_pre`;DP 两批重叠路径中 `state` 为属性访问的 `_StateDict`,边界张量严格只 pop 一次避免 idle DP rank 上 double-pop KeyError。

## 关键改动

```text
python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py: aiter 包装、aiter→Triton dispatch、跨层融合门控
python/sglang/srt/models/deepseek_v4.py: 三个 mHC 边界改走统一 dispatcher,处理 norm_fused=False 时补 layernorm/fp8 quant
test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py: 门控/dispatch 单测 + gfx95 数值等价测试
```

关键机制片段(dispatch 优先级):

```python
def try_mhc_fused_post_pre_boundary(...):
    aiter_result = try_aiter_fused_mhc_post_pre(...)
    if aiter_result is not None:
        return aiter_result
    triton_fn = hc_fn.T if fn_transpose else hc_fn
    return try_fused_hc_post_pre(...)
```

## 性能声明

PR 注明条件:模型 `deepseek-ai/DeepSeek-V4-Pro`,MI355X(gfx950)TP4,镜像 `rocm/sgl-dev:v0.5.15.post1-rocm720-mi35x-20260724`,env `SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`。benchmark_serving,ISL=8192 / OSL=1024,`--random-range-ratio 0.8`,`--ignore-eos`,每并发 10 prompts:

- concurrency 4:216.38→219.66 tok/s(+1.5%),TPOT 17.507→17.242ms;conc 8:+1.4%;conc 16:568.64→579.04 tok/s(+1.8%,峰值);conc 32:+1.0%;conc 64:1033.97→1042.04 tok/s(+0.8%)。TPOT 各并发均下降。

## 验证与回归风险

单测 `test_deepseek_v4_amd_fused_mhc.py` 8 passed(CPU 门控表、aiter 优先于 Triton、SM120 特例回归、attn 边界 fallback 必须先关上一层的 deferred post 等)。gfx95 数值测试:`next_residual` 对 unfused 参考 bit-exact,`layer_input`/`post_mix` bf16 容差内;`comb_mix` 有意不断言(fused/unfused 在生产 Sinkhorn 设置下原始值不同,正确性靠 E2E token match 建立)。E2E:GSM8K 1319 题 × 3 轮,baseline 均值 0.941 vs patched 0.945,invalid 全 0。风险:`norm_fused=False` 契约(Triton 融合路径不含 input/post-attention layernorm)需调用点补齐,PR 为此修了两个潜在静默腐蚀激活的 bug;aiter 失败会永久禁用融合路径走慢速 fallback;CI Extra 与 AMD ROCm 7.2 快照失败。

## 关联

同区域:既有 TileLang 与 Triton fused mHC post+pre 路径(本 PR 的 fallback 链)、FlashInfer `mhc_pre_big_fuse`。同分类其他 mHC PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
