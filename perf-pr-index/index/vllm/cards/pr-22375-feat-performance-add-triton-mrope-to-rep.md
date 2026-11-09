# PR #22375: [FEAT] [Performance] Add triton mrope to replace the torch code path

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tjtanaa |
| 合并 | 2025-08-09 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/22375 |

## 一句话结论

用 Triton 重写 mrope forward 替代 torch 路径，Qwen2.5-VL-7B 服务吞吐 2.20 → 2.24 req/s、Mean TTFT 快 3.9%。

## 动机

PR 正文 Purpose：优化 mrope forward pass，采用从 Liger-Kernel 的 `qwen2vl_mrope.py` 移植的 triton kernel，使其支持 vLLM 的 flatten input tensors，并支持形状为 `(3, num_tokens, head_dim // 2)` 的 cos/sin cache。正文提及与 issue #22293 相关。正文未说明改动前 torch 路径的具体耗时瓶颈。

## 优化点

在 `vllm/model_executor/layers/rotary_embedding/mrope.py` 新增 `@triton.jit _triton_qwen2vl_mrope_forward`：每个 program 处理一个 token，`pid` 定位 q/k 起始地址；按 `mrope_section_t`、`mrope_section_h` 把 head_dim 的 t/h/w 三段分别 mask 并累加得到 `cos_row`、`sin_row`，对应 cos/sin cache 的新 `(3, num_tokens, head_dim//2)` 布局。随后分别 load q、k 的左右半 head，按 `y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]` 完成旋转。新增 `benchmarks/kernels/benchmark_mrope.py` 生成 (3, num_tokens) 的 2D positions 测试数据并输出 CSV，以及 `tests/kernels/test_mrope.py` 覆盖 kernel 正确性。

## 关键改动

```text
vllm/model_executor/layers/rotary_embedding/mrope.py: 新增 _triton_qwen2vl_mrope_forward 替代 torch mrope forward，支持 flatten 输入与 (3, num_tokens, head_dim//2) cos/sin cache
benchmarks/kernels/benchmark_mrope.py: 新增 mrope kernel 基准脚本（Qwen2VL / Qwen2.5VL），结果写 CSV
tests/kernels/test_mrope.py: 新增 mrope kernel 测试
```

```text
@triton.jit
def _triton_qwen2vl_mrope_forward(q_ptr, k_ptr, cos, sin, num_tokens, ...):
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, head_dim // 2)
    half_hd = hd // 2
    t_cos = cos + pid * half_hd
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
```

## 性能声明

Kernel level improvement over torch implementation: ~2~3x（正文 Test Result，未注明该 kernel 级测试的硬件/精度条件）。lm_eval Qwen2.5-VL-7B-Instruct on chartqa：baseline explicit_prompt/anywhere_in_answer relaxed_correctness 0.8644 / 0.8644，After PR 0.8636 / 0.8636。Serving benchmark（ROCm：`VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 MIOPEN_FIND_MODE=FAST`，`Qwen/Qwen2.5-VL-7B-Instruct` TP=1，client `benchmark_serving.py --backend openai-chat --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --num-prompts 1000 --max-concurrency 64`）：Benchmark duration 453.55 → 447.35 s（1.4% faster），Request throughput 2.20 → 2.24 req/s（1.8% higher），Output token throughput 251.88 → 255.26 tok/s（1.3% higher），Total token throughput 459.85 → 466.12 tok/s（1.4% higher），Mean TTFT 4877.85 → 4688.14 ms（3.9% faster），Median TTFT 3353.20 → 3090.05 ms（7.8% faster），P99 TTFT 23555.70 → 21858.97 ms（7.2% faster），Mean TPOT 226.51 → 223.63 ms（1.3% faster）；Median TPOT 220.16 → 222.08 ms（0.9% slower）、Mean ITL 331.43 → 333.89 ms（0.7% slower）等项小幅变慢。

## 验证与回归风险

Test Plan 为 mrope kernel tests 与 lm_eval chartqa；Test Result 报告 "Kernel test: Passed"、kernel 级约 2~3x、以及上表 serving 对比。精度方面 lm_eval chartqa 由 0.8644 略降至 0.8636。回归风险：正文表格显示部分延迟指标（Median TPOT、Mean/Median ITL）有小幅劣化；正文未说明对其他模型、硬件或精度路径的影响。正文 Test Result 另附 mrope_benchmark_results_20250806_173029.csv 下载链接并声明 "Kernel test: Passed"；Checklist 中 Purpose、Test Plan、Test Result 三项勾选，文档更新项未勾选；正文还给出 ROCm serving 的 server 与 client 完整启动命令及 MIOPEN_USER_DB_PATH、MIOPEN_FIND_MODE、SAFETENSORS_FAST_GPU 等环境变量。正文在 Test Result 中说明 kernel 级结果另存为 CSV 文件，并给出服务基准的完整指标表，包含成功请求数、基准时长、各类吞吐以及 TTFT、TPOT、ITL 的均值、中位数与 P99 分项。

## 关联

无（正文仅提及关联 issue #22293，未给出前序/后续 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
