# PR #37205: [Kernel] Add gpt-oss Router GEMM kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-03-18 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready, ci/build, gpt-oss, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 1 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/37205 |

## 一句话结论

新增 gpt-oss Router GEMM kernel，batch size 1 下输出 token 吞吐 +1%-2%。

## 动机

PR 正文 Purpose 仅说明「This PR add gpt-oss optimized Router GEMM kernel」并给出 batch size 1 下 1%-2% 输出 token 吞吐提升，未详述改动前的具体瓶颈。其 Micro bench 显示低 batch size 下 PyTorch 路径吞吐明显偏低：gpt-oss-20b 在 batch_size=1 时 PyTorch 为 0.031421 TFLOPs、vLLM 新 kernel 为 0.089570 TFLOPs，gpt-oss-120b 为 0.123305 → 0.355313 TFLOPs；正文分别给出 openai/gpt-oss-20b 与 openai/gpt-oss-120b 两组 batch_size 1 至 8192 的吞吐表，并注明 "gpt_oss_router_gemm kernel has better throughput for low batch size"。

## 优化点

新增 CUDA kernel（csrc/moe/gpt_oss_router_gemm.cu / .cuh），经 moe_ops.h、torch_bindings.cpp、vllm/_custom_ops.py 注册为自定义 op。vllm/model_executor/layers/fused_moe/router/gate_linear.py 把原三层 GEMM dispatch 扩为四层：在 DSV3 specialized kernel 之后、cuBLAS bf16×bf16→fp32 之前插入 gpt-oss specialized kernel，条件为 SM90+、bf16 weight、batch<=128、且维度命中 num_experts [32,128] 与 hidden_size [2880]，否则逐级回退 cuBLAS 或 F.linear。vllm/model_executor/models/gpt_oss.py 及 lora 的 gate_linear / utils 同步适配。正文指出该 kernel 在低 batch size 下吞吐更好，高 batch size 时 vLLM 侧饱和在约 33 TFLOPs 而 PyTorch 继续增长。diff 还显示 csrc/moe/gpt_oss_router_gemm.cu 与 .cuh 为新增文件，moe_ops.h、torch_bindings.cpp、vllm/_custom_ops.py 负责把该 op 暴露给 Python 层，vllm/model_executor/models/gpt_oss.py 与 lora 侧的 gate_linear.py、utils.py 同步改用新的四层 dispatch。Micro bench 用 benchmarks/kernels/benchmark_router_gemm.py，Serving benchmark 用 vllm serve 与 vllm bench serve，Unit test 为 tests/kernels/moe/test_router_gemm.py。

## 关键改动

```text
csrc/moe/gpt_oss_router_gemm.cu / .cuh: 新增 gpt-oss Router GEMM CUDA kernel
csrc/moe/moe_ops.h、csrc/moe/torch_bindings.cpp、vllm/_custom_ops.py: 注册并暴露自定义 op
vllm/model_executor/layers/fused_moe/router/gate_linear.py: 四层 dispatch 与 gpt-oss kernel 命中条件
benchmarks/kernels/benchmark_router_gemm.py、tests/kernels/moe/test_router_gemm.py: 微基准与单元测试
vllm/model_executor/models/gpt_oss.py: 模型侧接入
```

```python
    # Dimensions supported by the gpt-oss specialized kernel
    GPT_OSS_SUPPORTED_NUM_EXPERTS = [32, 128]
    GPT_OSS_SUPPORTED_HIDDEN_SIZES = [2880]

        # gpt-oss specialized kernel eligibility (SM90+, exact dims)
        self.allow_gpt_oss_router_gemm = (
            self.weight.dtype == torch.bfloat16
            and current_platform.is_cuda()
            and is_hopper_or_blackwell
            and output_size in self.GPT_OSS_SUPPORTED_NUM_EXPERTS
            and input_size in self.GPT_OSS_SUPPORTED_HIDDEN_SIZES
        )
```

## 性能声明

PR 正文给出 "1% - 2% output token throughput improvement at batch size 1"。Serving benchmark（openai/gpt-oss-20b、--tensor-parallel-size 1、--max-num-seqs 16、--no-enable-prefix-caching，sharegpt、--sharegpt-output-len 300、temperature 0）：concurrency=1 时 Output token throughput 205.19 → 208.73 tok/s、Mean TPOT 4.78 → 4.70 ms；concurrency=16 时 648.16 → 651.26 tok/s。Micro bench（benchmark_router_gemm.py、--max-batch-size 8192）：gpt-oss-20b batch_size=1 PyTorch 0.031421 → vLLM 0.089570 TFLOPs，gpt-oss-120b 0.123305 → 0.355313 TFLOPs。测试条件：模型 openai/gpt-oss-20b（微基准另含 120b），硬件 PR 正文未注明（dispatch 需 SM90+/Hopper/Blackwell CUDA），精度 bf16（weight.dtype == torch.bfloat16），workload vllm bench serve sharegpt 与 router gemm 微基准。

## 验证与回归风险

新增单元测试 tests/kernels/moe/test_router_gemm.py，正文称 Unit test passed。精度用 gpqa（openai/gpt-oss-20b、--n-threads 200、--reasoning-effort low）：main score 0.5561868686868687、PR 0.5662878787878788；Accuracy Testing 段用 gpt_oss.evals 的 gpqa 任务给出 main 与 PR 两份完整评估报告及输出样例。回归风险：仅 SM90+ / bf16 / 指定维度 / batch<=128 走新 kernel，其余回退 cuBLAS 或 F.linear；高 batch size 下新 kernel 吞吐饱和（约 33 TFLOPs）不如 PyTorch；影响 MoE router GEMM、gpt-oss 模型与 LoRA gate_linear 路径。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
