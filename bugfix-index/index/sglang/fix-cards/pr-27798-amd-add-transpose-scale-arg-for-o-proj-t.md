# PR #27798: [AMD] Add transpose_scale arg for o_proj to fix GLM accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27798](https://github.com/sgl-project/sglang/pull/27798) |
| 作者 | 1am9trash |
| 合并日期 | 2026-06-17 |
| labels | amd |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
AMD gfx95 上 GLM-5.1 精度掉到约 0;o_proj 的 fp8 组量化漏传 `transpose_scale` 导致 scale 布局错配,补上该参数修复。

## 问题现象
0610 的 Nightly CI 发现 GLM-5 精度掉到约 0("GLM-5 accuracy dropped to ~0"),对应 issue(bingxche/sglang-ci-bot#99)。正文指明受影响的是 GLM-5.1,即 AMD 平台(aiter/gfx95 bpreshuffle)上走 MLA absorb 路径、o_proj 权重为 fp8 的场景。

## 根因
正文披露:sglang [PR#27289](https://github.com/sgl-project/sglang/pull/27289) 把 V4 的 scale-transpose element-wise kernel 融合进前置的 quant kernel,但在 `forward_mla.py` 中漏掉了 `o_proj` 的 scale 参数,造成 GLM-5.1 精度问题。aiter [PR#3041](https://github.com/ROCm/aiter/pull/3041)(已合并)为 `fused_flatten_fp8_group_quant` 增加了 `transpose_scale` 支持,但需要 sglang 侧同步 aiter 版本升级(正文注明 merge 前需要 aiter version bump,readiness 也一并更新)。

## 修复方案
改动在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` 的 `forward_absorb_core`:o_proj 权重为 `torch.float8_e4m3fn` 时调用 `fused_flatten_fp8_group_quant` 的两处分支,均补充 `transpose_scale=_use_aiter_bpreshuffle_gfx95` 参数:

```diff
                 elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                     attn_bmm_output = fused_flatten_fp8_group_quant(
-                        _bmm_buf, group_size=128, dtype_quant=torch.float8_e4m3fn
+                        _bmm_buf,
+                        group_size=128,
+                        dtype_quant=torch.float8_e4m3fn,
+                        transpose_scale=_use_aiter_bpreshuffle_gfx95,
                     )
```

即量化产生的 scale 按 gfx95 aiter bpreshuffle 的布局要求转置,与 o_proj GEMM 的预期对齐。

## 影响与回归风险
- 影响窗口:sglang PR#27289 融合 scale-transpose 之后、本修复之前的 AMD/gfx95 版本。
- 回归面:仅影响 forward_mla 的 absorb 路径中 o_proj fp8 量化分支(两处调用),且仅在 `_use_aiter_bpreshuffle_gfx95` 为真时行为变化;非 gfx95 或非 fp8 o_proj 路径不变。依赖 aiter 新版本提供 `transpose_scale` 支持。正文给出精度结果:GSM8K v4-pro 0.950、GLM-5.1 0.942。

## 关联
- [bingxche/sglang-ci-bot#99](https://github.com/bingxche/sglang-ci-bot/issues/99)(正文提到的 issue)
- [sglang PR #27289](https://github.com/sgl-project/sglang/pull/27289)(引入问题的 PR)
- [aiter PR #3041](https://github.com/ROCm/aiter/pull/3041)(提供 transpose_scale 支持)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
