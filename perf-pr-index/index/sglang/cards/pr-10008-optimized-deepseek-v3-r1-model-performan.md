# PR #10008: Optimized deepseek-v3/r1 model performance on mxfp4 run

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kkHuang-amd |
| 合并 | 2025-09-04 |
| 分类 | Quantization (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,约 8 个文件,含 2 个新文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/10008 |

## 一句话结论

DeepSeek-V3/R1 在 AMD gfx95 MXFP4(WMXFP4)运行下,把动态量化融合进 layernorm/GEMM/RoPE/flatten 各算子,端到端吞吐提升约 10%。

## 动机

WMXFP4 量化模型运行时,activation tensor 的动态 mxfp4 量化是独立算子,每次 layernorm、GEMM、bmm、flatten 之后都要单独跑一遍 quant kernel,产生额外访存与 kernel launch 开销。PR 将量化行为融合进上游算子,减少这些开销。

## 优化点

核心思路是"前一个算子直接输出已量化的 (x, x_s) 元组,下游直接消费"。具体包括:(1) `fused_rms_mxfp4_quant` 把 RMSNorm 与 mxfp4 量化合成一个 kernel,用于 input_layernorm 和 MLA 的 q_a/kv_a layernorm(可同时处理 q、k 两组);(2) `gemm_afp4wfp4_pre_quant` / `batched_gemm_afp4wfp4_pre_quant` 接收 bf16 输入,在 GEMM 内部完成量化,用于 MLA 吸收路径的 w_kc/w_vc 两个 bmm 和 router gate 的 `gemm_a16w16_atomic`;(3) `fused_qk_rope_cat` 把 RoPE 与 q/k concat 合并;(4) `fused_flatten_mxfp4_quant` 处理 o_proj 输入的 flatten+量化;(5) 引入 `gemm_output_zero_allocator`(BumpAllocator)预分配 router/shared-experts GEMM 的 fp32 零初始化输出缓冲。权重侧新增 `quark_post_load_weights`,对 mxfp4 的 kv_b_proj 权重做 w_kc/w_vc 拆分与重量化(含已量化 uint8 权重先反量化到 fp32 再拆分重量的路径)。所有新路径由 `SGLANG_USE_AITER` + `is_gfx95_supported()` 双条件门控,默认关闭。

## 关键改动

```text
python/sglang/srt/layers/communicator.py: prepare_attn 接受 quant_format,mxfp4 时用 fused_rms_mxfp4_quant 替代 input_layernorm。
python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py: 新文件,转发 aiter 的 fused_rms/flattened_mxfp4_quant、batched pre-quant GEMM。
python/sglang/srt/layers/rocm_linear_utils.py: 新文件,aiter_dsv3_router_gemm(M<=256 走 atomic GEMM + zero allocator)与 fused_qk_rope_cat。
python/sglang/srt/models/deepseek_v2.py: MLA absorb prepare/core 接入 fused layernorm、pre-quant bmm、fused rope+cat、fused flatten quant;model 层创建 gemm_output_zero_allocator。
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py: apply_weights 支持 (x, x_s, y) 元组输入与 fused quant GEMM。
python/sglang/srt/utils.py: 新增 is_gfx95_supported()。
```

## 性能声明

约 10% throughput 提升。测试条件(来自 PR 正文):服务端 `SGLANG_USE_AITER=1 python3 -m sglang.launch_server --model-path ams/DeepSeek-R1-WMXFP4-Preview --tp 8 --trust-remote-code --chunked-prefill-size 131072`;客户端 `bench_serving --dataset-name random --random-range-ratio 1 --num-prompt 500 --random-input 3200 --random-output 800 --max-concurrency 128`。前后对比以截图形式给出,正文未给出具体数值;硬件型号未注明(代码路径为 AMD gfx95 ROCm)。Accuracy Tests 一节为空,PR 未提供精度数据。

## 验证与回归风险

PR 未新增单元测试(Accuracy Tests、单测 checklist 均为空)。回归风险集中在:(1) 新路径仅在 `_use_aiter && gfx95` 时激活,但 deepseek_v2.py 中 `quant_format` 的判断直接比较 `weight == torch.uint8`(diff 原样),若实际生效依赖的是 weight dtype 判断分支,可能存在路径不触发或误触发的隐患;(2) MLA absorb 路径引入 tuple 型 hidden_states(携带已量化 x 与 scale),与非 tuple 分支的短路逻辑交错,其他 attention backend 组合需回归;(3) quark_post_load_weights 对已量化 uint8 权重的反量化-重拆分流程(mxfp4_to_f32/e8m0_to_f32)影响权重加载正确性;(4) gemm_output_zero_allocator 依赖 M<=256 才复用零缓冲,模型侧要求 n_routed_experts==256 且 embedding_dim==7168 才分配。

## 关联

同区域 AMD/ROCm DeepSeek 优化:PR #13617(ROCm DeepSeek R1 FP8 优化)、PR #21511(AMD FP8 KV cache/attention)、PR #23927(AMD FP8 MLA)。本卡批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
