# PR #37340: [Perf] Add tuned triton moe config for Qwen3.5 H200, 9.9% E2E throughput improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-03-18 |
| 分类 | MoE(major) |
| Label | performance, ready, qwen |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/37340 |

## 一句话结论

为 Qwen3.5 在 H200 新增 fp8_w8a8 block-wise fused MoE 调优配置,decode 热路径(bs128)E2E 输出吞吐 +9.9%。

## 动机

Qwen3.5 MoE(E=256,fp8_w8a8,block_shape=[128,128])在 H200 上没有专属 Triton fused MoE 配置,只能运行默认 tile 参数;同时 `benchmark_moe.py` 的架构解析不认识 Qwen3.5,无法为其生成配置。作者注明算力有限,只针对实践中最热的 bs=128 档做了调优。

## 优化点

两部分:1) `benchmarks/kernels/benchmark_moe.py` 泛化架构解析——`get_model_params` 改用 `getattr(config, "architectures", None)` 并兜底 `type(config).__name__`,新增 `Qwen3_5MoeForConditionalGeneration`/`Qwen3_5MoeTextConfig` 分支;新增 `resolve_dtype()`(ROCM 强制 fp16,否则依次取 config.dtype、text_config.dtype,最后回退 bfloat16)。2) 新增配置 JSON `E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128]`,覆盖 bs 1-4096 各档的 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`,标注 triton 3.6.0。

## 关键改动

```text
benchmarks/kernels/benchmark_moe.py: 支持 Qwen3.5 架构识别与 dtype 解析回退
vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json: 新增 H200 调优配置
```

## 性能声明

模型 Qwen/Qwen3.5-35B-A3B-FP8(启动参数含 --enable-expert-parallel,GPU 为 H200),`vllm bench serve --dataset-name random --random-input-len 2 --random-output-len 512 --request-rate inf --num-prompts 128`:输出吞吐 6313.91→6940.23 tok/s(request throughput 12.33→13.56 req/s,total token throughput 6338.57→6967.34 tok/s),Mean TTFT 306.44→169.15ms,Mean TPOT 19.69→18.11ms;标题口径为 9.9% E2E throughput improvement。精度:gsm8k 5-shot flexible 0.7801 / strict 0.7650(仅本 PR 单次结果,正文未附 main 对照)。

## 验证与回归风险

验证为 E2E serving benchmark + lm_eval gsm8k;benchmark 脚本有架构解析兜底,不破坏其他模型。回归面:配置仅命中 E=256、N=512、H200、fp8_w8a8、block_shape=[128,128] 的形状;收益数字针对 bs=128 档,其余 batch 档配置同批生成但未单独报告收益;Triton 版本标记 3.6.0。

## 关联

同分类 tuned config 类 PR:#27967(NemotronH H100/L40S)。素材未列出其他前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
