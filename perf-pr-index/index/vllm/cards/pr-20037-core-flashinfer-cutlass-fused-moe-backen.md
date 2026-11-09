# PR #20037: [Core] FlashInfer CUTLASS fused MoE backend (NVFP4)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wenscarl |
| 合并 | 2025-07-18 |
| 分类 | MoE (tier: major) |
| Label | performance, ready, ci/build, v1, deepseek |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files;diff 摘录另有 12 个文件因超长被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/20037 |

## 一句话结论

集成 FlashInfer NVFP4 CUTLASS fused MoE 内核(`VLLM_USE_FLASHINFER_MOE=1` 开启),DeepSeek-R1-FP4 在 B100x4 TP+EP 下吞吐约 1.9-2.9x。

## 动机

vLLM 此前对 ModelOpt NVFP4 量化 MoE 模型只有自研 `cutlass_moe_fp4` 路径:每次 GEMM 结果都重新分配中间张量、写法未接入 modular kernel 抽象,且缺少与 FlashInfer CUTLASS 内核的对接选项。FlashInfer 上游(flashinfer-ai/flashinfer#1113)提供了 NVFP4 CUTLASS fused MoE 内核,支持 DP+EP、TP+EP、DP+TP+EP 并行组合,值得作为可选后端接入。

## 优化点

两条线:(1) 新增 FlashInfer 后端——通过新环境变量 `VLLM_USE_FLASHINFER_MOE` + `use_flashinfer_cutlass_kernels` 属性门控,新建 `FlashInferExperts`(modular kernel 的 `FusedMoEPermuteExpertsUnpermute` 实现,调用 `flashinfer_cutlass_fused_moe`,传入 6 个 quant_scales,仅支持 nvfp4 w4a4)与 `FlashInferCutlassMoEPrepareAndFinalize`(prepare 时量化输入,DP 场景用 `all_gatherv` 按 DP 分块聚合 topk_weights/topk_ids/激活,再在通信后做 `fp4_swizzle_blockscale`);(2) 把 `cutlass_moe_fp4` 重构为 modular kernel 结构——拆出 `run_cutlass_moe_fp4` 与 `CutlassExpertsFp4`,`cutlass_fp4_moe_mm` 改为写入调用方提供的 out 张量,中间结果经 `_resize_cache` 复用 workspace13/workspace2,消除每次调用的中间分配;`apply` 接口统一增加 `extra_expert_args`,NVFP4 因需要两级动态 scale,通过 `skip_quant`/`skip_weight_reduce` 把量化逻辑移进 MoE body。注意:重构后的 `cutlass_moe_fp4` 后端支持 DP/TP/DP+TP,不支持 EP;FlashInfer 后端支持 EP 组合。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py: 新增 FlashInferExperts 模块化内核(NVFP4 w4a4)
vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py: 新增 prepare/finalize,DP all_gatherv + fp4 swizzle
vllm/model_executor/layers/fused_moe/cutlass_moe.py: cutlass_moe_fp4 重构为 run_cutlass_moe_fp4 + CutlassExpertsFp4,workspace 复用
vllm/_custom_ops.py: cutlass_fp4_moe_mm 改为写入外部 out 张量
vllm/envs.py + fused_moe/config.py: VLLM_USE_FLASHINFER_MOE 开关与 use_flashinfer_cutlass_kernels 门控
```

关键摘录(FlashInfer 调用的 scale 组装):

```python
quant_scales = [
    a1_gscale,
    w1_scale.view(torch.int32),
    g1_alphas,
    a2_gscale,
    w2_scale.view(torch.int32),
    g2_alphas,
]
_ = flashinfer_cutlass_fused_moe(
    hidden_states, topk_ids.to(torch.int), topk_weights,
    w1.view(torch.long), w2.view(torch.long),
    output_dtype=out_dtype, quant_scales=quant_scales,
    input_sf=a1q_scale, tp_size=self.tp_size, tp_rank=self.tp_rank,
    ep_size=self.ep_size, ep_rank=self.ep_rank, output=output,
)
```

## 性能声明

PR 正文 benchmark:`benchmark_throughput.py`,模型 nvidia/DeepSeek-R1-FP4,quantization=modelopt_fp4,input-len 1024 / output-len 1024,max-model-len 2048,TP size 4 + enable_expert_parallel,max-num-seqs 128,128 prompts,dummy weights,在 B100x4 上测试。

- eager 模式(v1 engine):关闭 0.34 req/s / 701.67 total tok/s / 350.99 output tok/s;开启 0.65 req/s / 1333.20 total tok/s / 667.08 output tok/s(1.9x,使用 pynccl 的 allgatherv 与 reduce-scatter);另注明 pynccl 较 native all2all 约 1.13x。
- graph 模式(仅 V0 engine,VLLM_USE_V1=0):关闭 0.56 req/s / 1138.12 / 569.06;开启 1.18 req/s / 2411.15 / 1208.55(正文标注 2.1x)。
- graph 模式(V1 engine):关闭 0.40 req/s / 825.88 / 413.37;开启 1.24 req/s / 2542.87 / 1271.43(约 3.1x,正文未给倍数标注)。

## 验证与回归风险

PR 正文未附测试命令或 accuracy 结果(Test Plan 部分未在素材中给出)。回归风险:`apply` 签名变化波及所有 modular kernel expert 实现(batched_deep_gemm、batched_triton_or_deep_gemm、deep_gemm、deepep_ht/ll prepare_finalize 等),`prepare`/`finalize` 接口同步增加 `extra_prepare_args`/`extra_finalize_args`;NVFP4 量化路径的 scale 语义(两级量化、swizzle 时序在通信前/后)是正确性敏感点;`use_batched_format=False` 下仅 Standard 激活格式。FlashInfer 后端仅在 `VLLM_USE_FLASHINFER_MOE=1` 且安装支持时生效,默认关闭。

## 关联

上游依赖 flashinfer-ai/flashinfer#1113;同区域为 fused MoE 后端(cutlass_moe_fp4、deep_gemm、triton、deepep 路径)。本卡所在批次素材中未列出其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
