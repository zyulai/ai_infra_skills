# PR #5194: Support Llama4 fp8 inference

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HandH1998 |
| 合并 | 2025-04-09 |
| 分类 | Quantization(tier: major) |
| Label | high priority, quant |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留大部分文件并注明 "3 file(s) dropped" |
| 链接 | https://github.com/sgl-project/sglang/pull/5194 |

## 一句话结论

为 meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 打通 fp8 推理:给 Triton fused MoE kernel 引入 per-channel 权重量化支持并统一 fp8/int8 scale 加载分支,GSM8K 93.6 / MMLU 86.2。

## 动机

支持 Llama-4-Maverick-17B-128E-Instruct-FP8 的 fp8 推理。该 checkpoint 的 expert 权重采用 per-channel(fp8 channel-wise)量化格式,而 SGLang 的 Triton fused MoE kernel 此前只在 block-wise 与 tensor-wise scale 下工作(fp8 分支与 int8 分支各自独立),缺 channel-wise 路径,无法直接加载运行。

## 优化点

(1) `fused_moe_kernel` 新增 `per_channel_quant` 编译期常量,并把原本散在 `use_fp8_w8a8` / `use_int8_w8a8` 两处的 scale 加载逻辑合并为一条三分支:block-wise(group_k/group_n)> 0 时按 block 加载;`per_channel_quant` 为真时按列加载 weight scale(b_scale_ptrs 含 `offs_bn`)且激活走 per-token scale;否则 tensor-wise 标量 scale。fp8 与 int8 共用同一结构。(2) `invoke_fused_moe_kernel` 在 block_shape 为 None 且权重 per-channel 量化时,动态激活量化改传 `use_per_token_if_dynamic=per_channel_quant`(`sgl_scaled_fp8_quant` / `vllm_ops.scaled_fp8_quant`);int8 路径补 assert:非 block-wise 时必须 channel-wise。(3) `per_channel_quant` 参数沿 `fused_moe → fused_experts → inplace/outplace_*` 全链路透传。(4) compressed_tensors config 解析并保存 `packed_modules_mapping`;fp8_utils / w8a8_fp8 / w8a8_int8 按权重量化策略传入 `per_channel_quant`;model_loader/weight_utils 适配加载;`mllama4.py` 处理 llama4 fp8 expert 与 dense 层不同的权重格式。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py: per_channel_quant 三分支 scale 加载 + 激活 per-token 量化
python/sglang/srt/layers/quantization/fp8_utils.py, w8a8_fp8.py, w8a8_int8.py: 按权重策略接线 per_channel_quant
python/sglang/srt/models/llama4.py, mllama4.py: Llama4 fp8 模型/expert 权重格式适配
test/srt/test_triton_moe_channel_fp8_kernel.py: 新增 channel-fp8 kernel 测试
```

## 性能声明

PR 正文 Benchmark(启动命令 `--model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tp 8`,硬件未注明):GSM8K(1400 题)93.6,MMLU 86.2。无吞吐/延迟数字;TODO 中自列 "Tune channel-fp8 triton fused moe kernel" 未勾选,即 kernel 尚未调优。

## 验证与回归风险

新增 `test_triton_moe_channel_fp8_kernel.py` 单测;checklist 的文档与 throughput/latency benchmark 项未勾选。风险面:fused MoE kernel 的 scale 分支重构影响所有 fp8/int8 w8a8 路径(block-wise DeepSeek 与 tensor-wise 原路径行为需靠既有单测回归保证);`use_per_token_if_dynamic` 改变了 CUDA/非 CUDA 两套动态量化入口的传参;compressed_tensors 新增 packed_modules_mapping 解析对既有 checkpoint 兼容(config 缺省为空 dict)。

## 关联

同区域:同分类 Quantization 的 PR #31109(删 QServe/FBGEMM)、Llama4 系列模型支持;TODO 提示后续 channel-fp8 Triton kernel 调优 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
