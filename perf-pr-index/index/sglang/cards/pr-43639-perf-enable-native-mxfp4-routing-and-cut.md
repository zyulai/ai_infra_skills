# PR #43639: [Perf] Enable native MXFP4 routing and CuTeDSL fusion for MiMo

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @leonardHONG |
| 合并 | 2026-10-14 |
| 分类 | MoE(major) |
| Label | run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/43639 |

## 一句话结论

MiMo MXFP4 改用 FlashInfer 原生路由 + 延迟 finalize 并接入 CuTeDSL 层边界融合,decode 吞吐最高 +11.76%。

## 动机

MiMo 的 MXFP4 路径此前有两个开销点:MoE 路由在 SGLang 侧单独计算(预计算 topk 再喂给 routed MoE kernel),且 MoE 输出在 AllReduce 通信之前就完成 finalize。既然已有 CuTeDSL 层边界融合基础设施,可以让 FlashInfer 原生完成 grouped routing、把 finalize 推迟到通信/归一化阶段一并做掉。

## 优化点

三条主线。其一,原生路由:为 `Mxfp4FlashinferTrtllmMoEMethod` 增加 `configure_bypassed_topk`,当 topk 配置满足 grouped topk + sigmoid scoring + correction_bias 且无 custom routing/共享 expert 时,把 `TopKOutputFormat` 置为 BYPASSED,`apply` 里改调 `trtllm_fp4_block_scale_moe`(传 routing_logits,`RoutingMethodType.DeepSeekV3`),由 FlashInfer 路由 kernel 直接算 topk,而不是先跑 SGLang topk 再调 `trtllm_fp4_block_scale_routed_moe`。其二,延迟 finalize:满足条件时 `forward_normal` 走 `forward_deferred_finalize`,返回 `MoeDeferredFinalize`(gated_shared_output=None,reduce 用 `sum_post_experts_output`),把 MoE finalize 与 AllReduce/RMSNorm 融合。其三,自动选择:`_flashinfer_allreduce_fusion_auto_enable` 对 MiMo 架构在 SM100+BF16+受支持 TP(2/4/8/16)+CP=1+非 PDMux 时默认选 cutedsl 后端;CuTeDSL workspace 增加 `include_shared_expert=False` 支持(MiMo 无 shared expert)。expert capture、分布记录、simulate routing、expert remapping 场景保留标准路由并排除在自动选择之外。

## 关键改动

```text
python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py:
  configure_bypassed_topk + trtllm_fp4_block_scale_moe 原生路由分支
python/sglang/srt/models/mimo_v2.py: BYPASSED topk 判定、deferred finalize、
  install_cutedsl_fusion(include_shared_expert=False)
python/sglang/srt/arg_groups/overrides.py: MiMo 自动启用 cutedsl 后端
python/sglang/srt/layers/flashinfer_mnnvl_cutedsl.py 与
  layers/layer_boundary/fusions/cutedsl.py: include_shared_expert 参数化
python/sglang/srt/layers/moe/topk.py: _eplb_remap_enabled 更名
  is_expert_remapping_enabled
```

## 性能声明

测试条件:MiMo-V2.6-Pro-RL(rev `73875d00b30a`),8x B300,TP8,并发 1,FA4,page size 128,KV 容量 32768,禁用 prefix cache;96 条冻结请求,8,000 输入/1,024 输出,每个版本各起一次新 server,3 次 warmup(EOS 仅计时阶段忽略):

- 有 DFlash:decode 302.44 -> 338.01 tok/s(+11.76%),E2E 285.44 -> 315.72 tok/s(+10.61%),TTFT 204.75 -> 216.63 ms;96 请求全部发生 partial rejection,acceptance 21.38% -> 21.61%,收益含验证负载差异。
- 无 DFlash:decode 176.54 -> 186.58 tok/s(+5.69%),E2E 170.75 -> 179.74 tok/s(+5.26%),两种模式 TTFT 平均增加约 12 ms。

PR 注明这是单次 fresh-server 对比,仅覆盖单并发负载,非多次运行的置信估计。

## 验证与回归风险

精度:128 道冻结 GSM8K 题,精确数值打分 123/128 -> 124/128(两升一降),严格字符串打分 119/128 -> 120/128;24 例 fixed-prefix 全词表 logprob 对比,22 例 argmax 一致,mean TVD 3.15%、最大 14.10%。PR 明确声明 FlashInfer 原生路由返回 BF16 expert weights(预计算路径为 FP32),输出与 DFlash acceptance 轨迹不逐位一致,不构成数值等价证明。风险:bypassed topk 仅在无 expert remap/无分布记录/非投机范围/无 a2a backend 时启用,但 `configure_bypassed_topk` 的契约判断(DeepSeekV3 路由方法)若与模型 gate 不匹配会直接 raise;`include_shared_expert=False` 改变了 CuTeDSL finalize preset 选择,对原有带 shared expert 的模型默认值保持 True,理论风险在早加载(shared load)路径。

## 关联

RoPE/KV-store fusion 在 #43106。同区域:#15464(MiMo V2 Flash FlashInfer 融合)。CuTeDSL allreduce fusion 亦被 #23402(MNNVL backend)等覆盖。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
