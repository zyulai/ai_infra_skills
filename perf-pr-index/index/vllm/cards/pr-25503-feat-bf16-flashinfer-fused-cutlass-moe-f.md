# PR #25503: feat: BF16 FlashInfer Fused Cutlass MOE for Hopper and Blackwell Expert Parallel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @djmmoss |
| 合并 | 2025-09-24 |
| 分类 | MoE(tier: major) |
| Label | performance, ready |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(5 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/25503 |

## 一句话结论

为 Hopper/Blackwell 的非量化(BF16/FP16)MoE 接入 FlashInfer Cutlass fused MoE kernel(EP、DP=1,经 `VLLM_USE_FLASHINFER_MOE_FP16` opt-in),以 Qwen3-Next-80B EP 场景验证。

## 动机

FlashInfer 的 Cutlass fused MoE(`FlashInferExperts`)此前仅支持 nvfp4 与 fp8 两种 quant dtype(构造函数 assert),非量化 BF16 MoE 只能走默认 Triton 路径;Hopper/Blackwell 上 expert parallel 场景(Qwen3-Next 等)缺少该 fused kernel 选项。

## 优化点

`FlashInferExperts` 接受 `quant_dtype=None`(非量化):`apply` 分为 fp8 / nvfp4 / none 三路,none 时无 quant scales、无 input scale、权重不做 `view(torch.long)`;新增 `flashinfer_cutlass_moe` 入口——以 `FUSED_MOE_UNQUANTIZED_CONFIG` 组装 `FusedMoEModularKernel`(prepare/finalize + `FlashInferExperts`);prepare 中 `nvfp4_block_scale_interleave` 仅在 nvfp4 时执行。`UnquantizedFusedMoEMethod` 的启用门控为:`has_flashinfer_cutlass_fused_moe()` 且 `VLLM_USE_FLASHINFER_MOE_FP16=1` 且 `use_ep` 且 `dp_size == 1` 且 CC ≥ 9;未启用时按缺失原因输出提示(EP 可用未开启 / DP 暂不支持)。`process_weights_after_loading` 在启用时把 `w13_weight` 换半排列为 `[w3; w1]`(kernel 期望布局);`forward_cuda` 新增对应分支。`WorkspaceBuffer.get` 对空 shape/None 直接返回 None。仅支持 silu activation(显式 assert)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py: FlashInferExperts 支持 quant_dtype=None;新增 flashinfer_cutlass_moe 入口
vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py: nvfp4 scale interleave 条件化
vllm/model_executor/layers/fused_moe/layer.py: 启用门控、forward_cuda 分支、w13 重排为 [w3; w1]
vllm/model_executor/layers/fused_moe/modular_kernel.py: WorkspaceBuffer.get 空 shape 返回 None
vllm/envs.py: 新增 VLLM_USE_FLASHINFER_MOE_FP16(默认 False,纳入 compute_hash)
```

## 性能声明

Hopper 性能结果仅以截图形式附在正文(无文字数字,本卡无法引用具体值);测试命令为 `vllm serve Qwen3-Next-80B-A3B-Instruct --enable-expert-parallel -tp 4 --max-num-batched-tokens 8192 --async-scheduling` 对比 Triton 与 FlashInfer 两种 MoE 后端。精度(GSM8K 5-shot,lm_eval,TP2 + EP,`VLLM_USE_FLASHINFER_MOE_FP16=1`,Qwen3-Next-80B-A3B-Instruct):flexible-extract exact_match 0.8506±0.0098,strict-match 0.8059±0.0109。

## 验证与回归风险

验证为 lm_eval GSM8K(如上);正文未声明单元测试。回归影响:默认关闭(opt-in);启用需 EP + DP=1 + CC ≥ 9 且 FlashInfer 具备 cutlass fused MoE;w13 换半布局仅在启用时执行,不影响其他路径;仅 silu activation,其他 activation 在启用时会 assert 失败;DP>1 场景明确不支持(仅提示日志)。

## 关联

同分类 MoE 区域的其他相关 PR 未在素材中列出;依赖 flashinfer 提供 cutlass fused MoE API。与 fp8/fp4 版本的 `VLLM_USE_FLASHINFER_MOE_FP8/FP4` 开关并列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
