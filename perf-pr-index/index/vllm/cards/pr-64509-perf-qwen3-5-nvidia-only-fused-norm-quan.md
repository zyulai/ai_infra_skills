# PR #64509: [Perf][Qwen3.5] (NVIDIA-only) Fused norm+quant and gate+quant

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-10-30 |
| 分类 | Quantization(major) |
| Label | ci/build, qwen, nvidia, quantization |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用但被截断(1 个文件未包含) |
| 链接 | https://github.com/vllm-project/vllm/pull/64509 |

## 一句话结论

Qwen3.5 NVIDIA 路径把 static FP8/NVFP4 量化融合进 Gemma add-RMSNorm、GDN gated norm 与 attention sigmoid gate,TPOT 最高降约 6.7%。

## 动机

Qwen3.5 的 NVIDIA no-compile 路径中,norm/gate 之后紧跟独立的 `scaled_fp8_quant` / `scaled_fp4_quant` kernel,量化和 norm 各自读写一遍激活。既有 norm+quant 融合工作(#56530、#61073、#45364 等)均未覆盖 Qwen3.5 的该路径。

## 优化点

四类融合:(1) 新 Triton kernel `gemma_add_rms_norm_quant` 把 Gemma add-RMSNorm 与消费者(qkv_proj / GDN in_proj_qkvz / dense 层 gate_up_proj)的 static FP8 或 NVFP4 输入量化合成一个 kernel,GDN 场景还可额外写出 bf16 normed 激活供 `in_proj_ba` 读取;(2) `fused_gdn_gated_rms_norm` 增加 FP8 输出选项;(3) `fused_sigmoid_gate_mul` 增加 FP8 输出选项(服务于 o_proj);(4) NVFP4 分支在 kernel 内直接写 128x4 swizzled block scale。派发按每次调用检查 `get_input_quant_key`:仅 static FP8(`kFp8StaticTensorSym`)或无 K padding 且 `K % 64 == 0` 的 `kNvfp4Dynamic` 走融合,block/per-token FP8、BF16、LoRA 等全部回退原路径。fused norm 无条件 `launch_pdl=True`,且 M≤128 时提前 `gdc_launch_dependents` 与消费者 GEMM 的权重加载重叠。

## 关键改动

```text
vllm/models/qwen3_5/nvidia/ops/fused_norm_quant.py: 新增 gemma_add_rms_norm_quant(FP8/NVFP4,含 PDL)
vllm/models/qwen3_5/nvidia/ops/fused_gdn_gated_norm.py: gated norm 增加 FP8 输出
vllm/models/qwen3_5/nvidia/model.py: _norm_quant_key 派发 + forward_quant,DecoderLayer 接线
vllm/kernels/triton/fused_sigmoid_gate_mul.py: 增加 static FP8 输出与 strided gate 回退
vllm/model_executor/models/qwen3_next.py: sigmoid gate + o_proj FP8 输入量化融合
```

## 性能声明

Microbenchmark(GB300,FlashInfer CUPTI,CUDA graph,冷 L2,5 次取中位):融合链比 unfused norm+quant 在 decode batch 尺寸快 1.7–2.0×,大 M 快 1.1–1.6×;如 attn in-norm+FP8(5120) M=1 4.64→2.69 µs(1.73×),gated norm+FP8(6144) M=1 3.81→1.89 µs(2.02×)。E2E:`nvidia/Qwen3.8-27B-NVFP4` 于 B300,random 8k/1k,TP1 TPOT c1/c16 -6.7%、c64 -2.2%;TP2 c1 -1.6%、c16 -2.4%、c64 -6.5%(最高 6.5%)。GSM8K 5-shot greedy 1319 题:0.9492 → 0.9545(噪声内)。

## 验证与回归风险

98 个测试在 B300 通过(`test_gemma_add_rms_norm_quant.py`、`test_fused_sigmoid_gate_mul.py`、`test_gdn_output_projection.py`、`test_qwen3_5.py`、`test_quant_activation_contract.py`);新增 Buildkite `(B200) Qwen3.5` 步骤;附 dispatch 审计矩阵(dummy-weight、MoE、LoRA、MTP、layer 0 等)。回归风险:per-token / 128x128 block FP8 不在覆盖范围,走回退;warp-dispatch 规则仅在 GB300(SM103)扫过,B200 可能需重扫;PDL 对抗性前驱在 M=128 最多 +4%;FP8/FP4 code 允许 1 step 内偏差(不保证跨架构 bitwise)。

## 关联

互补/相关:#56530(TP>1 的 AR+norm+quant)、#61073(通用 RMSNorm 同契约)、#45364 / #45855 / #42469 / #61783 / #59553(其他 norm+quant 融合)、#41021、#41457;CUDA graph 归因不涉及。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
