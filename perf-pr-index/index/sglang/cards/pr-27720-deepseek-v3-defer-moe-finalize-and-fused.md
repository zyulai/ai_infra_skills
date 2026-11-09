# PR #27720: [DeepSeek V3] Defer moe finalize and fused it with main stream add

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kpham-sgl |
| 合并 | 2026-06-13 |
| 分类 | MoE (tier: major) |
| Label | deepseek, run-ci, jit-kernel, bypass-fastfail |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,6 个文件,含 3 个新文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/27720 |

## 一句话结论

DeepSeek/Kimi 类 MoE decode 时推迟 expert finalize 并把 shared expert 加法融进同一 CUDA kernel,每层省约 2us,TPOT 提速 1-2%。

## 动机

优化 Kimi K2.5 NVFP4 的 decode 性能。原 `DeepseekV3MoE.forward` 双流执行时,finalize(topk 加权汇总)由 flashinfer trtllm MoE 内部完成,随后还要单独跑 routed 输出与 shared expert 输出的 add(以及可能的 routed_scaling_factor 乘法),多出独立 kernel 与一次完整访存;PR 注明 prefill 阶段尚未受益(仍是 2 次独立 launch),本 PR 只优化 decode 形态。

## 优化点

两处机制改动:(1) 交换双流分工——主流执行 routed experts、alt stream 执行 shared experts;(2) 主流对 flashinfer trtllm fp4 MoE 传 `do_finalize=False`,把 `gemm2_out / expert_weights / expanded_idx_to_permuted_idx` 以 `FlashInferTrtllmDeferredFinalizeOutput` 透传出来,等 alt stream 汇合后用新 JIT kernel `moe_finalize_fuse_shared` 一次完成 "sum_k w_k * gemm2_out[permuted_idx(t,k)] + shared_output"。该 kernel fork 自 flashinfer 的 finalizeKernel/finalizeKernelVecLoad,输出 bf16,支持 PDL(SM>=90,`cudaGridDependencySynchronize`/`cudaTriggerProgrammaticLaunchCompletion`),提供 general 与 128-bit 向量化两种变体,按块数是否 >=1184(148 SM x 8)启发式分流;topk 权重支持 fp32/bf16 两种 dtype。触发条件:`SGLANG_ENABLE_MOE_DEFERRED_FINALIZE`(默认关)+ flashinfer_trtllm runner backend + `ModelOptNvFp4FusedMoEMethod` + BYPASSED topk 格式 + shared_output 存在。

## 关键改动

```text
python/sglang/jit_kernel/csrc/moe/moe_finalize_fuse_shared.cu: 新增,fused finalize+shared-add CUDA kernel(PDL,bf16 输出,MAX_TOPK=64)。
python/sglang/jit_kernel/moe_finalize_fuse_shared.py: 新增,JIT 加载与 Python 封装。
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: do_finalize=False 透传,contextvar 开关,finalize_flashinfer_trtllm_deferred_output。
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: supports_deferred_finalize 判定与 forward_deferred_finalize 入口。
python/sglang/srt/models/deepseek_v2.py: 双流分工交换,deferred 分支替代 maybe_fuse_routed_scale_and_shared_add。
python/sglang/srt/environ.py: 新增 SGLANG_ENABLE_MOE_DEFERRED_FINALIZE(默认 False)。
```

## 性能声明

每层时间 -2us;TPOT 提速 1-2%。测试条件(PR 正文):Kimi K2.5 NVFP4,TP=8,ISL 220,负载 80k;前后对比为 profiler 截图,未给出表格数值。decode 生效,prefill 仍为原两次 launch。

## 验证与回归风险

PR 声称已加单测(checklist 勾选),但 pack 的 diff 中未包含测试文件(diff 只含上述 6 个源码文件);CI 状态:Base Test 未通过(no_entry_sign),Extra Test 失败(pack 中标注)。回归风险:(1) diff 中自带 FIXME:flashinfer 按 routing_logits dtype(fp32)分配 expert_weights 缓冲却写入 bf16,PR 用 `view(torch.bfloat16)` 重解释,依赖上游修复(flashinfer issue #3595),上游行为变化会破坏该假设;(2) topK 上限 64、vec-load 路径要求 hiddenDim/hiddenDimPadded 均为 8 的倍数,否则回落 general kernel;(3) PDL 依赖 SM>=90;(4) 默认关闭(env gate),只影响 flashinfer trtllm + NVFP4 量化 + BYPASSED topk 的 decode 路径。

## 关联

同区域 MoE finalize 融合:PR #46724(muse-spark-open:MoE finalize 与 2shot_push all-reduce 融合,本批次内);PR #18612(fuse silu-mul into NVFP4 expert)。本卡无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
