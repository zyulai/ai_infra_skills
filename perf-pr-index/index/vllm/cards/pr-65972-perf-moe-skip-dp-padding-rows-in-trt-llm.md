# PR #65972: [Perf][MoE] Skip DP padding rows in TRT-LLM monolithic MoE with vLLM's top-k

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-11-09 |
| 分类 | Speculative Decoding (tier: major;classify 输出,内容实为 MoE kernel 路径) |
| Label | ready, nvidia, verified |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数为 16 个文件(diff 完整,未见截断) |
| 链接 | https://github.com/vllm-project/vllm/pull/65972 |

## 一句话结论

DP 下 TRT-LLM monolithic MoE 改用 vLLM top-k 预计算路由、把 padding 行标 -1 跳过:大专家模型 DP+EP 吞吐最高 +26–27%,padding 在 decode 步占 34–65%、混合步占 37–84%。

## 动机

DP 下每个 rank 把 batch pad 到全 rank 最大 batch。FlashInfer TRT-LLM monolithic MoE 是 Blackwell 上 block-FP8/MXFP8 checkpoint 的默认 kernel,它从 router logits 自行路由,padding 行像真 token 一样选中专家并被加载、计算。既有的 `VLLM_MOE_SKIP_PADDING` 无法生效——它靠 vLLM router 给 padding 行写 expert id -1,而 monolithic kernel 根本不调用 vLLM router。padding 量很大:decode 步(每 rank 最多 32 行)34–65% 是 padding;混合(EDA prefill+decode)步每个 rank 都 pad 到跑 prefill 的 rank,37–84% 是 padding。

## 优化点

机制上把"kernel 从 logits 自己路由"换成"vLLM top-k 预计算路由 + kernel 跳过 -1 行":DP 下 runner 先调 `router.select_experts`(int32 top-k),padding 行置 -1,再走 monolithic 的 precomputed-routing op(`trtllm_fp8_block_scale_routed_moe`)。配套改动:`FusedMoEExpertsMonolithic` 实现 `supports_precomputed_routing`/`apply_precomputed_routing`(TRT-LLM block FP8 与 MXFP8),与 logits 调用共享的 kwargs 收进 `_block_scale_kwargs` helper;NaiveDPEP 的 prepare/finalize 新增 `prepare_routed`,all-gather 的是 top-k id/weight 而非 logits;router 不自行 mask padding 的(如 DeepEP v2 之外的 grouped top-k)由 runner 在 top-k 后补 mask;`routed_scaling_factor` 只作用一次。生效条件全部为:SM100 + block-FP8/MXFP8、DP>1(EP 默认 allgather_reducescatter,或无 EP)、无 EPLB、无 sequence parallel、kernel 支持的 routing 方法、`VLLM_MOE_SKIP_PADDING` 开、routing simulator 关;决策只依赖 DP size,各 rank 路径一致。不做 batch size 阈值(top-k 启动是每层固定 5–15us,阈值会漏掉 padding 最多的混合步);`VLLM_MOE_SKIP_PADDING=0` 退回原行为。autotune cache key 在 DP+routed 时加新字段,避免旧缓存命中后跳过 tuning。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py: apply_precomputed_routing 走 trtllm_fp8_block_scale_routed_moe
vllm/model_executor/layers/fused_moe/runner/moe_runner.py: DP 下选择 routed 路径并补 padding mask
vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py: prepare_routed dispatch top-k 而非 logits
vllm/model_executor/layers/fused_moe/modular_kernel.py: apply_monolithic_routed / apply_routed 接口
vllm/model_executor/warmup/flashinfer_autotune_cache.py: cache key 增加 moe_routed 因子
```

```python
if (self.routed_experts.quant_method.is_monolithic
        and self.moe_config.dp_size > 1
        and self.routed_experts.supports_precomputed_routing
        and not isinstance(self.router, RoutingSimulatorRouter)
        and (is_padding := get_moe_padding_mask(...)) is not None):
    topk_weights, topk_ids = self.router.select_experts(
        hidden_states=hidden_states, router_logits=router_logits,
        topk_indices_dtype=torch.int32, input_ids=input_ids)
    if not self.router.masks_padding:
        topk_ids = topk_ids.masked_fill(is_padding.unsqueeze(1), -1)
    fused_out = self.routed_experts.forward_monolithic_routed(
        hidden_states, topk_weights, topk_ids)
```

## 性能声明

GB200,FlashInfer 0.7.3,`vllm bench serve` ShareGPT temperature 0,全部启用 speculative decoding(MTP 3 / EAGLE3 3),每侧各启动两次。输出吞吐:GLM-5.3-FP8 DP8 在 c8/16/64 为 +8.6%/+25.9%/+26.6%(如 c64 2315/2425→2975/3026 tok/s,TPOT −27.0%);MiniMax-M3-MXFP8 DP4 为 +13.1%/+4.0%/+5.0%(c4 501/582→612/613 tok/s,TPOT −13.6%);Qwen3.8-Flash-Next-FP8(4.9M 参数小专家,DP4)吞吐全部在噪声内(+2.1%/+1.2%/+5.5%/−0.7%),但大 decode 步变慢:c64 33–128 行 +6.4%,c256 33–128 行 +13.2%、129–512 行 +6.6%。kernel 单调用基准(GB200,CUDA graph,5×50 replays 中位数):0% padding 时 top-k 路径慢 +1% 到 +11%,约 20% padding 打平,75% padding 时 GLM 最多 −553us。GSM8K(1319 题,5-shot)三模型精度不变,差异最大 0.016 且在两侧自身重复波动内。TP4 下强制开启无收益(±2.4% 内噪声),故路径仅在 DP 启用。

## 验证与回归风险

新增单测:`test_trtllm_fp8_monolithic_routed_skips_padding[block/mxfp8/block-grouped]`(同一量化输入下 routed op 与 kernel 自路由在真实行相等、padding 行输出全零,覆盖 softmax top-k / sigmoid+bias / DeepSeek-V3 grouped)、`test_runner_routes_dp_padding_rows_to_no_expert[router masks/runner masks]`、`test_runner_prepares_monolithic_logits[dp 1/4]`(simulator 仍走 logits)、`prepare_routed` dispatch 断言、autotune key 矩阵。GB200 上相关 MoE 测试 1345 passed, 2 skipped, 0 failed。风险面:Qwen 类小专家模型大 decode 步退化(已给 opt-out);MTP drafter 的 MoE 不标 padding(保持 kernel 自路由);NVFP4 monolithic 在 DP+EP 下不被选中,走已 mask 的 modular 路径;GLM-5.3 autotune on 在 main 上即崩溃(既有问题,另行跟踪),本 PR 测试时两侧均关 autotune;升级后 DP block-FP8/MXFP8 部署首次启动会 retune 一次。

## 关联

同区域:#50759(draft,modular 路径 router mask padding,与本 PR 无冲突)、#54109(DeepEP v2 padded format)、#57946(b12x padding sentinel)——均为其他 padding 处理路径。后续方向:flashinfer-ai/flashinfer#9000(FlashInfer 侧 padding mask,可免去 top-k 启动,runner 可直接传 mask)。pack 内无其他同分类 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
