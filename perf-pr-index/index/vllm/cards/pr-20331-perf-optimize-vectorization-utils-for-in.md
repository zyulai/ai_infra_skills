# PR #20331: [Perf] Optimize Vectorization Utils for Int 8 Quantization Kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-07-04 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/20331 |

## 一句话结论

给 Int8 量化 kernel 补对齐向量化读,H100 上 w8a8 Llama-3.1-8B E2E 吞吐 +3%。

## 动机

PR 正文标注修复 issue #20327。Int8 量化 kernel 在计算 per-row absmax 与 min/max 时用标量循环逐元素读取输入,未利用已有的对齐向量化工具。PR 通过扩展 vectorization utils,让这类"只读输入做归约"的循环也能走向量化路径。

## 优化点

在 `vectorization_utils.cuh` 的 `vectorize_with_alignment` 前段增加"整段已对齐"fast path:当地址按 `WIDTH = VEC_SIZE * sizeof(InT)` 对齐且长度是 `VEC_SIZE` 整数倍时,直接按 `vec_n_t` 向量读写并提前返回,跳过原有 prefix / 主区 / tail 三段处理(注释说明当前输出与输入一致,故不检查输出对齐)。新增只读版 `vectorize_read_with_alignment`(含 `DefaultReadVecOp` 把向量拆成标量回调,以及仅需 `scalar_op` 的重载),同样带对齐 fast path 与 unaligned prefix / 主向量区 / tail 标量回退,并 `static_assert` VEC_SIZE 为 2 的幂。随后把 `int8_quant_kernels.cu` 中 `dynamic_scaled_int8_quant_kernel` 的 absmax 循环与 `dynamic_scaled_int8_azp_quant_kernel` 的 MinMax 循环改为 `vectorize_read_with_alignment<16>` 调用,仍由原有 `cub::BlockReduce` 完成归约。

## 关键改动

```text
csrc/quantization/vectorization_utils.cuh: 核心。vectorize_with_alignment 增加整段对齐 fast path;新增 vectorize_read_with_alignment 只读模板、DefaultReadVecOp 与单 scalar_op 重载
csrc/quantization/compressed_tensors/int8_quant_kernels.cu: dynamic_scaled_int8_quant_kernel 的 absmax 循环、dynamic_scaled_int8_azp_quant_kernel 的 MinMax 循环改用 vectorize_read_with_alignment<16>
```

关键代码(absmax 归约由标量循环改为向量化读):

```cuda
vectorize_read_with_alignment<16>(
    row_in, hidden_size, tid, stride, [&] __device__(const scalar_t& src) {
      const float v = fabsf(static_cast<float>(src));
      thread_max = fmaxf(thread_max, v);
    });
```

## 性能声明

PR 正文标注 "Performance (3% E2E throughput gain)",测试条件 H100,命令 `vllm bench throughput --model RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 --load-format dummy --input-len 1000 --output-len 100 --max-model-len 32768`(PR 正文未注明精度字段)。Main:37.24 req/s、40917.63 total tokens/s、3723.77 output tokens/s;37.42 req/s、41122.36 total tokens/s、3742.40 output tokens/s。Now(本 PR):38.38 req/s、42177.65 total tokens/s、3838.44 output tokens/s;38.55 req/s、42355.99 total tokens/s、3854.67 output tokens/s。Accuracy(`lm_eval` gsm8k 5-shot,同模型 `max_model_len=32768`):Main 与 Now 均为 flexible-extract 0.7733±0.0115、strict-match 0.7604±0.0118,无变化。

## 验证与回归风险

PR 正文仅给出上述 throughput benchmark 与 gsm8k 精度对比,未列出专门的单元测试命令。风险面:改动位于共享的 `vectorization_utils.cuh`(影响所有调用 `vectorize_with_alignment` 的量化 kernel 的对齐 fast path 行为)与 compressed_tensors Int8 量化 kernel 的归约路径;正文以 gsm8k 精度不变作为回归验证,未列其他模型或硬件的测试。

## 关联

无(PR 正文仅标注修复 issue #20327,未给出其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
