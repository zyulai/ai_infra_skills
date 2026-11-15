# PR #66799: [Perf][MoE] Mask DP padding rows in grouped top-k routing by default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-11-14 |
| 分类 | MoE(tier: major) |
| Label | ready, verified |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66799 |

## 一句话结论

Grouped top-k router(DeepSeek-V3/V3.2、GLM-4.5+/5、Kimi-K2/K3、MiMo-V2 等)默认给 DP/CUDA-graph padding rows 专家 id −1(与 fused top-k router 一致),modular MoE kernel 由此跳过它们:GLM-5.3-NVFP4 DP8 吞吐 +5% 至 +21%、MiMo-V2.6-Flash DP4 +5% 至 +7%、DeepSeek-V3.2-NVFP4 DP4 +3% 至 +8%,GSM8K 不变;不新增选项(`VLLM_MOE_SKIP_PADDING=0` 可关)。

## 动机

`GroupedTopKRouter` 只在 `skip_padding` 置位时把 runner 的 `is_padding` 传给 kernel,而 `FusedMoEFactory` 只对显式 `skip_padding=True`(Kimi-K3)且用 DeepEP v2 的模型置位(#56079)。其余 grouped top-k 模型把 padding rows 当真实 token 路由,modular kernel 为它们做全部计算。#65972 已对 TRT-LLM block-FP8/MXFP8 monolithic kernel 做了同样的事,本 PR 把默认行为扩展到 grouped top-k。

## 优化点

- `GroupedTopKRouter.masks_padding` 改为 `skip_padding or not rocm_aiter_ops.is_fused_moe_enabled()`(对齐 `FusedTopKRouter` / `FusedTopKBiasRouter`);mask 本身就是 opt-in 下已实现的机制:CUDA kernel 的 `is_padding` 或其余路径的 `masked_fill`。ROCm AITER 上 opt-in 语义不变(作者未确认 AITER experts 接受 −1)。
- `get_moe_padding_mask` 在 router 看到的行数超过本地 mask 覆盖范围时返回 `None`(naive DP dispatch 在路由前 all-gather 所有 DP rank 的行,本地 mask 不描述这些行;`topk_softmax` 与 `grouped_topk` CUDA kernel 都检查 `is_padding.size(0) == num_tokens`,传入会 raise)——这些行照旧无 mask 路由;该守卫同样适用于 fused top-k routers。
- `skip_padding`/`VLLM_MOE_SKIP_PADDING` 的 docstring/注释更新。
- 影响面:所有 `use_grouped_topk=True` 且非退化分组的 MoE 模型(DeepSeek-V2/V3/R1/V3.2、Kimi-K2/K2.5/K3、GLM-4.5/4.6/4.7/5/5.3、MiMo-V2、Hunyuan、EXAONE-MoE、LFM2-MoE、OpenPangu、Nemotron-H MoE 及 `n_group`/`topk_group` 的 Bailing/Ling、Sarvam 等)在 CUDA/XPU/ROCm(非 AITER fused MoE)的 modular kernel 上有 padding rows 时。不受影响:DeepSeek-V4/V4.1(自有 routing)、LongCat-Flash、plain softmax/sigmoid top-k(已被 fused top-k routers mask)、CPU、ROCm AITER、从 logits 路由的 monolithic kernels。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py: masks_padding 属性默认开启;_compute_routing 用 masks_padding 取 padding mask
vllm/model_executor/layers/fused_moe/router/base_router.py: get_moe_padding_mask 在 mask 行数不足时返回 None
vllm/model_executor/layers/fused_moe/router/router_factory.py: skip_padding 文档更新
vllm/model_executor/layers/fused_moe/layer.py: FusedMoEFactory 的 skip_padding 文档更新
vllm/envs.py: VLLM_MOE_SKIP_PADDING 注释更新
tests/kernels/moe/test_grouped_topk.py: 测试泛化为 skip_env × skip_padding × use_fused;新增 rows-gathered-before-routing 守卫测试
```

```python
     @property
     def masks_padding(self) -> bool:
-        return self.skip_padding
+        return self.skip_padding or not rocm_aiter_ops.is_fused_moe_enabled()
@@ -346,7 +346,7 @@ def valid_grouping() -> bool:
         is_padding = (
-            get_moe_padding_mask(router_logits.shape[0]) if self.skip_padding else None
+            get_moe_padding_mask(router_logits.shape[0]) if self.masks_padding else None
         )
```

## 性能声明

(各模型 Baseline = main 80e0cd68f8,After = 同树强制开启 grouped top-k padding masking;每侧 2 次 launch,顺序 Baseline/After/After/Baseline;ShareGPT,temp 0,`vllm bench serve`,4 prompts per unit of concurrency)

- **GLM-5.3-NVFP4**(`FLASHINFER_TRTLLM` NVFP4,DP8 + EP,2 nodes,MTP 3):c8 +5.4%、c16 +17.6%(第二 session +13.1%)、c64 +6.7%、c128 +16.6%、c256 +21.1%(均"both runs better";tok/s 如 c256 6484/6642 → 7922/7973)。
- **MiMo-V2.6-Flash-RL**(`FLASHINFER_TRTLLM_MXFP4_MXFP8`,DP4 + EP):c4 +2.6%(within noise)、c16 +5.8%、c64 +5.1%、c256 +6.8%。
- **DeepSeek-V3.2-NVFP4**(`FLASHINFER_TRTLLM` NVFP4,DP4 + EP):c4 +7.7%、c16 +3.4%、c64 +3.1%、c256 +1.5%(within noise)。
- **TPOT**:GLM-5.3-NVFP4 c8–c256 中位 TPOT −5.8% 至 −22.8%;MiMo −1.8% 至 −7.3%;DeepSeek-V3.2 −0.9% 至 −4.3%。
- **唯一变慢 bucket**:DeepSeek-V3.2 c256 的 decode steps(每 rank 33–128 rows)两次 run 均 +5.6%(占 step 时间 13%);该并发 throughput 在噪声内。
- **准确度**:GSM8K(1,319 题,5-shot)三模型 c4 与 c256 均在 launch 间波动内(GLM 0.908/0.905/0.902/0.895 vs 0.905/0.899/0.901/0.899;MiMo c4 0.920/0.917 vs 0.919/0.920;DeepSeek-V3.2 c4 0.958/0.955 vs 0.955/0.955 等),per-question 答案一致率与同版本重跑相当。

## 验证与回归风险

- 单元测试(GB200):`test_grouped_topk.py / test_routing.py / test_routing_simulator.py / test_topk_softplus_sqrt.py` 2800 passed/10 skipped/0 failed;另 `test_flashinfer.py -k "routed or grouped"`、`test_routed_experts_capture_monolithic.py`、`test_naive_dp_ep_dbo.py` 81 passed。
- `test_grouped_topk_masks_padding[skip_env × skip_padding × use_fused]`:mask 开时 padded rows 得 −1/weight 0(fused 与 unfused),env 关时不 mask,real rows 不变——main 上无 model flag 的四个 case 失败。`test_grouped_topk_skips_mask_for_rows_gathered_before_routing`:行数超过 mask 时路由完成且不 mask——main 上两 case 失败(CUDA kernel size check 或 `masked_fill` shape mismatch),去掉长度检查后在本分支也失败。
- 风险/限制:sequence-parallel MoE(EP + TP>1 + DP>1)下每 TP rank 路由自己的 chunk 但 `is_padding` 是全 batch mask,`[:num_tokens]` 让 TP>0 的 rank 拿到第一个 chunk 的 mask——padding 总在 batch 末尾,只会漏 mask 不会错 mask real token(fused top-k routers 在 main 上同有限制),mask re-sharding 留作 follow-up;Kimi-K3 未实测(其已验证布局无 DP padding,或用 DeepGEMM MegaMoE 自读 `is_padding`);仅 ShareGPT 测过,DeepEP LL/HT 未测;端到端只测了 GLM/MiMo/DeepSeek-V3.2 三模型,其余模型依赖论证(fused top-k 已向同样的 modular experts/all2all backends 传 −1,EPLB mapping kernel 把 −1 映射到 −1)。

## 关联

- #56079(DeepEP v2 的 skip_padding opt-in,本 PR 默认化的来源)、#65972(TRT-LLM block-FP8/MXFP8 monolithic kernels 的同款默认)、#50759(draft,generic torch.where mask,非重复——不排除 ROCm AITER、无 rows-gathered 守卫)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
