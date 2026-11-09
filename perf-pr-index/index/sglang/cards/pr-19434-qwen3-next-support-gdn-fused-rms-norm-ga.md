# PR #19434: [Qwen3-Next] Support gdn fused_rms_norm_gated

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-02-27 |
| 分类 | Speculative Decoding(classify 分类;实际作用于 Qwen3-Next GDN norm) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/19434 |

## 一句话结论

Qwen3-Next GDN 的 norm+gate 换用 Triton 融合 kernel `FusedRMSNormGated`,TTFT 提速 4%,吞吐同量级提升。

## 动机

attention/state-space 类架构(Qwen3-Next 的 GDN、Kimi 的 KDA)每层每步都要做 norm + gate 两次访存,融合成一个 kernel 可显著省带宽;`head_v_dim` 通常 <= 256,正落在为 D <= 512 优化的融合 kernel 最佳区间。此前 `layer_norm_gated_fwd_kernel` 只放在 KDA 的 `kda.py` 里,GDN 的 `qwen3_next.py` 用的是未融合的 `RMSNormGated`,且该 kernel 本质上是 KDA/GDN 共用的公共模块,应放到 fla 公共目录。

## 优化点

新增 `fla/fused_norm_gate.py`(改编自 flash-linear-attention 的 `fused_norm_gate.py`),把 (x + residual) -> norm -> gate 融合为单次 kernel 调用,前向分两个变体:D <= 512 用 `layer_norm_gated_fwd_kernel`,每个 program 处理 BT=32 行 x BD 列的 tile(面向吞吐);D > 512 用 `kernel1`(每 program 一行,降低寄存器压力与复杂度)。kernel 以 `IS_RMS_NORM`/`HAS_WEIGHT`/`HAS_BIAS`/`HAS_RESIDUAL`/`ACTIVATION`(swish/silu/sigmoid)等编译期开关统一 RMSNorm/LayerNorm 变体,gate 以 `y * g * sigmoid(g)`(swish)内联。GDN 侧 `qwen3_next.py` 把 norm 的选择改为:开启 piecewise cuda graph 时保留 `RMSNormGated`,否则用新的 `FusedRMSNormGated`。同时做代码搬迁:`kda.py` 中原来的 kernel 与 `FusedRMSNormGated` 删除,统一从新模块导入,`kimi_linear.py` 相应改导入路径。

## 关键改动

```text
python/sglang/srt/layers/attention/fla/fused_norm_gate.py: 新增融合 kernel
  (layer_norm_gated_fwd_kernel D<=512 tile 版 / kernel1 单行版)与 FusedRMSNormGated
python/sglang/srt/layers/attention/fla/kda.py: 删除内置 kernel,改为导入公共模块
python/sglang/srt/models/qwen3_next.py: GDN norm 按 piecewise cuda graph 开关
  在 RMSNormGated / FusedRMSNormGated 间选择
python/sglang/srt/models/kimi_linear.py: 导入路径迁移
```

## 性能声明

测试条件:`Qwen/Qwen3-Next-80B-A3B-Thinking`,TP4/DP4 + dp-attention,NEXTN 投机(steps 3, topk 1, draft tokens 4),禁用 radix cache;客户端 `bench_serving` random 数据集,200 prompts,输入 8000 / 输出 1500。PR 结论:融合 kernel 获得 4% TTFT 提速,吞吐同量级提升。对比数据:Mean TTFT 23,169.69 ms -> 22,252.74 ms;Request throughput 3.25 -> 3.39 req/s;Total token throughput 15,314.80 -> 15,959.30 tok/s;Mean E2E 36,279.77 -> 34,760.77 ms;Mean TPOT 17.76 -> 17.14 ms;accept length 3.83 -> 3.82(基本不变)。

## 验证与回归风险

精度:gsm8k(Qwen3-Next-80B-A3B-Instruct,lm_eval local-completions,5-shot)flexible-extract exact_match 0.8643±0.0094、strict-match 0.8461±0.0099,作者称 "no drops"。Checklist 均未勾选,pack diff 未含新增单测。风险:GDN norm 的实现按 `enable_piecewise_cuda_graph` 分叉,piecewise 路径仍用旧 `RMSNormGated`,两条路径数值/行为可能不完全一致;融合 kernel 对 D > 64KB(特征维)会直接 raise;KDA/Kimi linear 的导入重构影响所有引用 `fla.kda.FusedRMSNormGated` 的调用方。激活仅支持 swish/silu/sigmoid。

## 关联

同区域:#16397(layernorm-gated for Qwen3-Next)、#17094(GDN decode 优化)、#19775(fused GDN gating)、#21019(Qwen3.5 GDN split/reshape/cat 融合)、#30216(CPU 侧 gated norm 融合)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
