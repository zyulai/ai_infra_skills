# PR #33506: [Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amitz-nv |
| 合并 | 2026-02-12 |
| 分类 | MoE(tier: major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/33506 |

## 一句话结论

FlashInfer TRT-LLM fused MoE 支持 relu2 非门控激活(FP8/NVFP4)与 DeepSeek 路由,覆盖 Nemotron 类模型。

## 动机

Nemotron 系列模型使用非门控激活(relu2 / act-and-mul),而 FlashInfer trtllm fused MoE 的 Python 侧判定此前只接受 gated SiLU:_supports_no_act_and_mul() 返回 False、_supports_activation() 只认 SILU,导致这类模型的 FP8/NVFP4 权重无法落到 trtllm fused MoE 后端。FP8 per-tensor 路由白名单也缺 DeepSeekV3 路由类型。

## 优化点

把 activation_type 显式传入 FlashInfer trtllm fused MoE FP8/NVFP4 调用(新增 activation_to_flashinfer_int 映射:SILU→Swiglu、RELU2_NO_MUL→Relu2 等);NVFP4 权重准备按 is_gated 区分 gemm1_intermediate_size(非门控不乘 2),跳过 w13→w31 重排,并在非门控时使用 min_alignment=128 的中间维对齐 padding(新增 align_fp4_moe_weights_for_fi);FP8 per-tensor 权重旋转/scale 登记按 is_gated 分支;_supports_activation 放宽到 SILU+RELU2_NO_MUL;FP8 路由白名单加入 RoutingMethodType.DeepSeekV3;测试改用比例式 check_accuracy 并参数化 relu2_no_mul。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py: 非门控/relu2 能力放开;activation_type 传参;DeepSeekV3 路由
vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py: NVFP4 非门控权重准备与 128 对齐 padding
vllm/model_executor/layers/quantization/utils/flashinfer_utils.py: activation_to_flashinfer_int、align_fp4_moe_weights_for_fi
vllm/model_executor/layers/quantization/modelopt.py: apply_monolithic 支持集放宽
tests/kernels/moe/test_flashinfer.py: check_accuracy + relu2 参数化
```

## 性能声明

PR 未给出加速比数字。附 lm_eval GSM8K(5-shot,limit 500)正确性验证:Nemotron 3 Nano(VLLM_USE_FLASHINFER_MOE_FP8=1,VLLM_FLASHINFER_MOE_BACKEND=latency),flexible-extract 0.568 / strict-match 0.848。注意素材中命令为 NVFP4 模型名而日志显示加载 NVIDIA-Nemotron-3-Nano-30B-A3B-FP8,以 PR 原文为准记录,不做推断。属能力启用验证,非性能对比。

## 验证与回归风险

tests/kernels/moe/test_flashinfer.py:per-tensor FP8 no-graph 测试参数化 SILU/RELU2_NO_MUL,容差从 assert_close(atol=5.5e-2, rtol=1e-2) 改为比例判定(atol=0.1, rtol=0.85, percent=0.925),仅限 Blackwell(has_device_capability(100))。风险:padding 修改 layer.intermediate_size_per_partition;非门控路径的 scale 组合(g1_scale_c = ones/a2_scale 等)较绕;cutlass 路径测试同样放宽容差。

## 关联

同分类 MoE 卡:#46518(MXINT4 gated SiLU 判定放宽,同一批 FlashInfer MoE 能力判定)、#45703(Marlin MoE padding)、#20037(FlashInfer cutlass fused MoE backend)。无前序/后续 PR 声明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
