# PR #28892: Add TRTLLM MoE NVFP4 kernel to CompressedTensorsW4A4MoeMethod

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Victor49152 |
| 合并 | 2025-11-21 |
| 分类 | MoE（tier: major） |
| Label | performance, ready, nvidia |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/28892 |

## 一句话结论

让 compressed-tensors W4A4 NVFP4 MoE 也能走 FlashInfer TRT-LLM fp4 kernel：单 GB200 请求吞吐 +23.5%、TEP=2 +81.3%（对比 flashinfer.cutlass）。

## 动机

TRT-LLM FP4 MoE kernel 此前只接入 modelopt 量化路径；compressed-tensors（llm-compressor 产出 checkpoint 的主流路径）的 `CompressedTensorsW4A4MoeMethod` 只能用 flashinfer cutlass MoE，在 GB200 上错失更快的 TRT-LLM fp4 fused MoE kernel。同时两套路径各自维护 TRT-LLM 的权重 shuffle 与调用逻辑，存在重复。

## 优化点

- 把 TRT-LLM fp4 MoE 的调用抽成共享函数 `flashinfer_trtllm_fp4_moe`（flashinfer_fp4_moe.py）：输入 `fp4_quantize`（非 swizzled scale layout）、按 custom_routing_function 判定 Llama4 routing、DeepSeekV3 routing 时 router_logits 转 fp32，再调 `flashinfer.fused_moe.trtllm_fp4_block_scale_moe`。
- 权重准备函数 `prepare_static_weights_for_trtllm_fp4_moe` 从 `ModelOptNvFp4MoEMethod` 的方法抽为模块级共享函数：对每个 expert 做 w3/w1 permute 重排与 `nvfp4_block_scale_interleave` scale interleave（epilogue_tile_m=128）。
- `CompressedTensorsW4A4MoeMethod` 接入：`__init__` 经 `get_flashinfer_moe_backend()` 选后端；TRT-LLM 分支在 `process_weights_after_loading` 生成 shuffled weights/scales 参数与 `g1_scale_c`（= `w2_input_scale_quant * g1_alphas`）并删除不再使用的原始权重；`apply` 直接返回共享 helper，`maybe_make_prepare_finalize`/`get_fused_moe_quant_config` 返回 None（kernel 自带 prepare/finalize）。input global scale 归约按后端区分：TRT-LLM 用全局 min expand 到 num_experts，cutlass 路径按 dim=1 min（原实现为 max(dim=1)）。
- `modelopt.py` 的 `apply` 改调共享 helper，删除内联的约 130 行调用代码。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py: 新增共享 prepare_static_weights_for_trtllm_fp4_moe + flashinfer_trtllm_fp4_moe
vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py: W4A4 MoE 接入 TRT-LLM 后端分支
vllm/model_executor/layers/quantization/modelopt.py: 复用共享 helper，删除内联实现
```

## 性能声明

（GB200，checkpoint RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4；workload：random 300 in / 200 out、单图 (512,512,1)、1250 prompts、`--ignore-eos`、`--max-model-len 32768`、`--enable-expert-parallel`、`--async-scheduling`。）

- 单 GB200：Request throughput 28.00→34.58 req/s（TRT-LLM kernel 比 flashinfer.cutlass **快 23.5%**）；Output tok/s 5,599.97→6,916.91；Mean TTFT 16,529.75→13,332.46 ms；Mean TPOT 57.15→46.46 ms。
- TEP=2（expert parallel）：Request throughput 49.68→90.07 req/s（**快 81.3%**）；Output tok/s 9,935.25→18,013.18；Mean TTFT 8,506.92→2,727.64 ms；Mean TPOT 57.87→39.57 ms；Total tok/s 24,838.14→45,032.94。

Accuracy（GSM8K 5-shot，`--limit 0.5`，TP2 + EP）：compressedtensor + flashinfer trtllm fp4 moe flexible-extract 0.9606（与 compressedtensor + cutlass 的 0.9606 持平，高于 compressedtensor + flashinfer cutlass 的 0.9545）；modelopt + trtllm fp4 0.9348 与 modelopt + flashinfer cutlass 0.9348 持平。compressed-tensors checkpoint 由 llm-compressor 对 Qwen3-VL-235B-A22B 生成，modelopt 路径用 nvidia/Qwen3-235B-A22B-FP4。

## 验证与回归风险

- 验证：正文列出 5 条代码路径（modelopt/cutlass、modelopt/trtllm、compressedtensor/cutlass、compressedtensor/flashinfer cutlass、compressedtensor/trtllm fp4）均跑了 GSM8K 对照；diff 未新增单元测试。
- 回归风险：TRT-LLM 分支删除原始 w13/w2 权重与 scale（`del layer.*`），路径不可逆；input global scale 归约方式随路径变化（min 取全局或 per-dim），对 cutlass compressed-tensors 路径的数值行为有变（原为 max(dim=1)），PR 以 GSM8K 结果佐证；依赖 FlashInfer TRT-LLM kernel 可用性与 `get_flashinfer_moe_backend()` 选择逻辑；`routed_scaling_factor` 由原 modelopt 内联调用的 1.0 改为共享 helper 的 None。

## 关联

- 前序：modelopt 路径已有的 TRT-LLM fp4 MoE 接入（本 PR 将其抽为共享函数并扩展到 compressed-tensors）。
- 同区域：本批 MoE/量化 PR #61457（per-layer fused shared expert）、PR #22211（topk softmax fused kernel）、PR #32520（SM100F FP4 量化 kernel）、PR #45739（NVFP4 scale buffer）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
