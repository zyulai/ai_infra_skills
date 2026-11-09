# PR #20645: [NVIDIA] Add SM100 Flashinfer MoE blockscale fp8 backend for low latency

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kaixih |
| 合并 | 2025-07-19 |
| 分类 | MoE |
| Label | performance, ready |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),下方 diff 仅供内容参考 |
| 链接 | https://github.com/vllm-project/vllm/pull/20645 |

## 一句话结论

为 SM100 接入 FlashInfer 新的 blockwise-scaled FP8 fused MoE 后端,面向低延迟场景的 DeepSeek 类模型。

## 动机

FlashInfer 在 flashinfer#1212 中为 block-wise scaled FP8 引入了一个新的 MoE 后端(`trtllm_fp8_block_scale_moe`),目标是低延迟推理。此前 vLLM 的 Fp8MoEMethod 在 SM100 上走 triton/deep_gemm 路径,缺少这个 FlashInfer 后端选项。本 PR 把该后端接入 vLLM 的 fused MoE 框架。

## 优化点

机制上分三步:第一,把原先单一的环境开关 `VLLM_USE_FLASHINFER_MOE` 拆分为 `VLLM_USE_FLASHINFER_MOE_FP8` 与 `VLLM_USE_FLASHINFER_MOE_FP4`(后者沿用原 CUTLASS FP4 路径),新增 `has_flashinfer_moe()` 能力探测与 `flashinfer_trtllm_fp8_block_scale_moe` 懒加载封装。第二,新增 custom op `flashinfer_fused_moe_blockscale_fp8`(带 fake impl、`needs_fixed_stride_order` 标签,兼容 torch.compile):内部先用 `per_token_group_quant_fp8` 对激活做 128x128 group 量化,转置 scale 后连同 routing logits/bias 一次性交给 FlashInfer 内核,由其内部完成路由与两级 GEMM,因此启用该路径时跳过 vLLM 侧的 `FusedMoE.select_experts`,消除独立的 topk 内核与中间物化。第三,根据 FlashInfer 对 gate/up 两半激活作用位置与 vLLM 相反的约定,`process_weights_after_loading` 中用 `_swap_w13_to_w31` 交换 w13 权重与 scale。tile 大小按 `num_tokens*top_k/num_experts` 向上取 2 的幂并 cap 到 8-64。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: 新增 flashinfer_fused_moe_blockscale_fp8 custom op 与 tile 尺寸估算
vllm/model_executor/layers/quantization/fp8.py: Fp8MoEMethod 接入 flashinfer 路径,w13→w31 权重交换
vllm/envs.py: 拆分 VLLM_USE_FLASHINFER_MOE 为 FP8/FP4 两个开关
vllm/utils/flashinfer.py: trtllm_fp8_block_scale_moe 懒加载封装与 has_flashinfer_moe 探测
vllm/model_executor/layers/quantization/modelopt.py: FP4 侧开关改名引用
```

## 性能声明

PR 正文未给出 benchmark 数字,仅说明 FlashInfer 新后端面向 low latency(低延迟)场景;标题亦标注 "for low latency"。PR 未注明测试条件。

## 验证与回归风险

- PR 未新增测试文件(diff 中未见测试改动)。
- 约束较多:仅支持 DeepSeek 式路由(renormalize + grouped topk + sigmoid scoring、`routing_method_type=2`)、`block_shape == [128, 128]`、`top_k <= 8`、`topk_group <= 4`、专家数整除 4 与分组数等,断言失败即报错。
- 回归风险点:env 开关改名会影响已用 `VLLM_USE_FLASHINFER_MOE` 的用户脚本;`process_weights_after_loading` 的 w13 交换逻辑只在 flashinfer 分支生效,不触及其他后端。

## 关联

- 同为 SM100 FlashInfer MoE 系列的前序 PR:#20037(Core: FlashInfer CUTLASS fused MoE backend,即本 PR 拆分后的 FP4 路径)。
- 相关上游 PR:flashinfer-ai/flashinfer#1212。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
