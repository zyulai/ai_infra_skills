# PR #47354: [Perf] [NVIDIA] glm5: Route small-M BF16 GEMMs to FlashInfer warp split-K with PDL on SM107

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @akhilg-nv |
| 合并 | 2026-11-06 |
| 分类 | Speculative Decoding(classify 分类;实际作用于 GLM-5 小 M BF16 GEMM 路由) |
| Label | quant, deepseek, run-ci, bypass-fail-fast |
| Diff | pack 未提供 diffstat(正文注明 runtime 代码 +74/-8、3 文件,测试 +234、1 文件),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/47354 |

## 一句话结论

SM107 上把 GLM-5 TP4 decode 的 16 个小 M BF16 GEMM 键改走 FlashInfer warp split-K(PDL),decode 每输出 token 耗时 -2.51%。

## 动机

SM107(Vera Rubin NVL72)上 GLM-5 TP4 + EAGLE decode 每步只有 1-20 个 token,o_proj、fused_qkv_a、DSA indexer 的 fused wk + weights_proj、shared expert 的 gate_up/down 这些稠密 BF16 GEMM 当前分别走 CuTe DSL TGV kernel、cuBLAS(shared expert 与 indexer 常是 split-K GEMM + 独立 reduce kernel)或 fused_a JIT(M<=16 的 fused_qkv_a)。这个尺寸下每个 GEMM 仅几微秒、主要在搬权重,启动段能否与上一个 kernel 重叠与 kernel 本体同等重要。FlashInfer 的 warp split-K BF16 GEMM(主分支已为 Muse Spark keys 使用)以 PDL(programmatic dependent launch)启动:上个 kernel 还没结束就启动、发出权重加载,仅在真正读激活前等待——带 PDL 时这些形状下每次调用都比现路由快,不带 PDL 则在 o_proj 慢于 TGV。

## 优化点

仅 SM107 生效的纯路由改动(无 kernel 修改)。其一,在 split-K 启用且 `device_sm == 107` 时,`initialize_bf16_gemm_config` 把 `_active_warp_splitk_tactics` 重绑为 base 表 + 16 个 GLM-5 TP4 decode 键(M=1/4/5/20,EAGLE 4-1-5 的 draft/verify 尺寸;verify M=5x batch),`use_bf16_splitk_gemm`/`_bf16_splitk_gemm_out` 读取该活动表,两张原表都不改动;每个键用 FlashInfer 默认 tactic(M=1/5 的 88-110 个有效 tactic 原位 sweep 无一快 1.5% 以上)。其二,新增 `prefer_bf16_splitk_over_fused_a`,`prepare_qkv_latent` 据此把 fused_qkv_a 在 M=1/4/5 时从 fused_a JIT kernel 挪到 split-K 路由(q_b_proj 保留 fused_a)。其三,`precompile_splitk_tactics` 支持键子集,`precompile_sm107_splitk_tactics` 只编译 SM107 键,由 `GlmMoeDsaForCausalLM.precompile_kernels_after_loading` 在 CUDA graph capture 前调用。保留现路由的例外:indexer wq_b(M=5 无收益,M=20 与 q_b_proj 共形且后者慢 12.5%)与 shared down M=20(+8.7%)。`SGLANG_ENABLE_BF16_SPLITK_GEMM=0` 仍可整体关闭。

## 关键改动

```text
python/sglang/srt/layers/quantization/unquant.py:
  _BF16_SM107_WARP_SPLITK_TUNED_TACTICS(16 键)、_active_warp_splitk_tactics
  重绑、prefer_bf16_splitk_over_fused_a、precompile_splitk_tactics(keys)
python/sglang/srt/models/deepseek_v2.py: prepare_qkv_latent 接入
  prefer_bf16_splitk_over_fused_a
python/sglang/srt/models/glm4_moe.py: precompile_kernels_after_loading 预编译
test/registered/unit/layers/quantization/test_unquant_sm107_splitk.py:
  CPU 单测(SM107 vs SM100/SM103 路由、fused_qkv_a 条件、调用点)
```

## 性能声明

主测试:Vera Rubin NVL72 单 tray(4 GPU),GLM-5.3-NVFP4,TP4,AgentX GLM-5 TP4 c1 低延迟配置(cutedsl bf16 backend、禁 shared experts fusion、EAGLE 4-1-5、HiCache、FP8 KV cache、max-running-requests 8、模拟 acceptance 3.33),order base/PR/PR/base 各 900 s,125 请求配对:每请求 decode 时间/输出 token 中位数 -2.51% [95% CI -2.80, -2.16](两对分别 -2.54%/-2.47%,A/A 噪声 ±0.14%);interactivity(1000/p90 ITL)+2.6%(在 ±2.5-2.9% 噪声内);TTFT 无变化。静态负载(random 8K in/1K out)提示性结果:TPOT 并发 1 约 -3%、并发 4 约 -2%。Per-call decode profile: o_proj M=5 5.79 -> 4.93 us,fused_qkv_a M=5 4.51 -> 3.55 us,indexer wk M=5 4.48 -> 2.88 us(main stream);shared gate_up M=5 8.35 -> 4.10 us、down M=5 4.10 -> 2.82 us(side stream);cuBLAS split-K reduce kernel 消失(每 decode cycle 2,135 -> 2,034 kernels)。PDL 微基准:无 PDL 时 FlashInfer 在 o_proj 慢于 TGV,带 PDL 后 4.69-4.70 us。

## 验证与回归风险

精度:GSM8K(sgl-eval,300 题)98.00% -> 98.00%;`--max-running-requests 1`(每步都走新键)98.33% -> 98.33%。GPQA-Diamond 在 GB300(SM103 强制启用同键,test-only switch)90.40% -> 91.92%,配对差 +1.52 [-1.52, +4.55],在预设 5 分非劣效界内(仅排除粗大破坏)。16 个键逐一与 FP32 参考做逐元素容差检查(PDL 开/关,32/32 通过),SGLang 路由输出与直接 FlashInfer 调用 bit-equal,注入错误 64/64 被检出,PDL 链测试 1,600 次 replay 0 错,compute-sanitizer 0 错。CPU 单测覆盖 SM107/SM100/SM103 路由分叉与条件反转。风险:键按 shape 匹配,SM107 上任何模型只要 BF16 GEMM 形状完全一致就会走同一路由;CI 无 SM107 runner,新路由本身只在 NVL72 手工验证中运行;正文披露 FlashInfer warp split-K kernel 存在三处 PTX 内存模型未保证的同步顺序(已有 flashinfer#8080/#8081 修复,与本 PR 无关但共享该 kernel)。

## 关联

主分支已有 `_BF16_WARP_SPLITK_TUNED_TACTICS`(Muse Spark keys)作为 base 表。同区域:#27053(GLM5 BCG 与 prefill 使能)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
