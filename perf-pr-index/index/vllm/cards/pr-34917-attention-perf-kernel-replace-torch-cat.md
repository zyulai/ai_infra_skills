# PR #34917: [Attention][Perf][Kernel] Replace torch.cat with vectorized CUDA kernel MLA query concat - DeepSeek-V3.2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-03-09 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready, ci/build, v1, deepseek, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/34917 |

## 一句话结论

用向量化 CUDA kernel 替代 torch.cat 拼接 MLA query，峰值 ~11.8x，B300 上 TTFT 改善约 7%。

## 动机

PR 正文未设独立动机段落，只给出 "Speedup improves with the number of tokens, peaking at ~11.8× w.r.t. torch.cat"，说明原 torch.cat 拼接 ql_nope 与 q_pe 的开销随 token 数增长；正文开头注明 "PR #35105 needs to be merged first"。

## 优化点

新增 csrc/concat_mla_q.cuh 的 ConcatMLAQKernel：按 warp 对应一个 token×head，用 128/256-bit 向量（st128_cs/st256_cs、ld128_cs/ld256_cs）批量拷贝 ql_nope，再拷贝 64 维 rope（q_pe）。csrc/cache_kernels.cu 增加 concat_mla_q 入口，校验 nope_dim 为 512 倍数、rope_dim==64、三者 stride(2)==1，并按 total_warps/8 起 grid（每 block 8 个 warp / 256 线程）；当前仅实例化 NOPE_DIM=512（DeepSeek V3.2 MLA，rope dim 硬编码 64）。经 csrc/torch_bindings.cpp 与 vllm/_custom_ops.py 注册自定义 op，flashmla_sparse.py 的调用点改用新 op 替代 torch.cat；另加 benchmarks/kernels/bench_concat_mla_q.py、tests/kernels/test_concat_mla_q.py 与 .buildkite/test_areas/kernels.yaml 测试区域。新增的 bench_concat_mla_q.py 与 test_concat_mla_q.py 分别覆盖非连续 nope 输入（transposed BMM output）的微基准与单元测试，.buildkite/test_areas/kernels.yaml 把该文件纳入 kernels 测试区域；正文 Accuracy 段用 tests/evals/gsm8k/gsm8k_eval.py 对比 MAIN 与 PR。

## 关键改动

```text
csrc/concat_mla_q.cuh: 新增向量化 ConcatMLAQKernel（按 warp 处理 token×head）
csrc/cache_kernels.cu: 新增 concat_mla_q CUDA 入口与维度/stride 校验
csrc/torch_bindings.cpp、vllm/_custom_ops.py: 注册并暴露 op
vllm/v1/attention/backends/mla/flashmla_sparse.py: MLA query concat 调用点改用新 op
benchmarks/kernels/bench_concat_mla_q.py、tests/kernels/test_concat_mla_q.py: 微基准与测试
```

```cpp
#pragma unroll
  for (int i = 0; i < nope_vec_loads; i++) {
    const int offset = i * 32 + lane_id;
    if constexpr (use_256b) {
      st256_cs(reinterpret_cast<u32x8_t*>(nope_dst) + offset,
               ld256_cs(reinterpret_cast<const u32x8_t*>(nope_src) + offset));
    } else {
      st128_cs(reinterpret_cast<int4*>(nope_dst) + offset,
               ld128_cs(reinterpret_cast<const int4*>(nope_src) + offset));
    }
  }
```

## 性能声明

微基准（正文 "Perf eval on B300"，非连续 nope 输入 / transposed BMM 输出，concat_mla_q-transposed，单位 us）：num_tokens 8 时 torch.cat 5.055390 → concat_mla_q 1.892126；8192 时 3101.767921 → 355.234556；正文称 "Speedup improves with the number of tokens, peaking at ~11.8× w.r.t. torch.cat"。Serving：vllm serve nvidia/DeepSeek-V3.2-NVFP4 -tp 4，vllm bench serve（--input-len 16000 --output-len 1 --num-prompts 128 --num-warmups 20 --max-concurrency 64）：MAIN Mean TTFT 2394.86 ms → PR 2246.02 ms，即 "~7% TTFT improvement"；Request throughput 23.77 → 24.85 req/s，Total token throughput 380282.51 → 397582.64 tok/s。gsm8k eval：Accuracy MAIN 0.923 → PR 0.926。测试条件：硬件 B300（正文标题），模型 nvidia/DeepSeek-V3.2-NVFP4，精度 NVFP4，workload 为 concat 微基准与上述 vllm bench serve。

## 验证与回归风险

新增 tests/kernels/test_concat_mla_q.py 与 benchmarks/kernels/bench_concat_mla_q.py，并在 .buildkite/test_areas/kernels.yaml 登记测试区域；正文给出 gsm8k 精度对比（0.923 → 0.926）。回归风险：kernel 限定 nope_dim 为 512 倍数、rope_dim==64、且 ql_nope/q_pe/q_out 的 stride(2)==1，仅实例化 NOPE_DIM=512，属 DeepSeek V3.2 MLA 专用；影响 MLA/DSA sparse 后端的 query concat 路径；正文注明 PR #35105 需先合并。

## 关联

PR #35105（正文注明需先合并）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
