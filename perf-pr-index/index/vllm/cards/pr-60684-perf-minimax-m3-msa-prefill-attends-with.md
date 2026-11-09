# PR #60684: [Perf][MiniMax-M3] MSA prefill attends with the E4M3 query; bump fmha_sm100 to 968061a

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-10-11 |
| 分类 | Quantization (tier: major) |
| Label | ready, ci/build, nvidia, minimax |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/60684 |

## 一句话结论

MiniMax-M3 MSA prefill 直接用 E4M3 query 进 fmha_sm100 内核,免去每次 prefill 的 BF16 反量化拷贝,FP8 KV 下逐层加速 26-37%。

## 动机

开启 CUTLASS MSA sparse decode(NVFP4 KV cache,或 FP8 E4M3 cache 配 `CUTLASS_MSA` decode)时,fused QK-norm/RoPE/insert kernel 只写出 E4M3 格式的 query;`MiniMaxM3SparseMSAImpl` 却在每次 prefill 前把它反量化回 BF16 query buffer,而 `fmha_sm100` 的两个 prefill 内核本就原生接受 E4M3 query,这次反量化是纯开销。此外 vllm-project/MSA 的旧 pin(`be4a52a`)尚不含 Blackwell prefill 移植(在 packed K|V pages 上服务 E4M3 query,MSA#32)。

## 优化点

prefill 路径不再反量化:有 `query_fp8` 时 `qp = q_fp8[nd:]` 直接传入 sparse_atten kernel,`q_scale` 折入 `softmax_scale`(`self.scale * q_scale_float`);无 query_fp8(BF16 cache 或 FP8 cache 默认 Triton decode)时行为不变,仍传 BF16 query。同时把 `fmha_sm100` 依赖从 `be4a52a` bump 到 `968061a`:新 pin 包含 MSA 的 Blackwell prefill 移植,`sparse_atten_func` 接受 E4M3 query 在 FP8 E4M3 K/V 上跑 FP8 QK/PV MMA,`sparse_atten_nvfp4_kv_func` 在 E4M3 query 下把 NVFP4 `code × block_scale` 反量化到 E4M3(而非 BF16)再用;K 全局 scale 施加于 score、V 全局 scale 在 combine 中处理。cmake install 列表新增 `_jit_cache.py`、`msa_warmup.py`、`q8kv4_prefill_adapter.py` 与 `prefill_q8kv4/`。仅 Triton decode 回退仍读反量化后的 query。

## 关键改动

```text
vllm/models/minimax_m3/nvidia/sparse_attention_msa.py: prefill 的 qp 与
  softmax_scale 改为 E4M3 q 直传 + q_scale 折入
cmake/external_projects/fmha_sm100.cmake: pin be4a52a → 968061a,新增 install 文件
vllm/models/minimax_m3/nvidia/model.py: 注释更新(仅 Triton decode 反量化)
tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py:
  新增/改写 prefill query dtype 断言与 E4M3-staged 参考实现
```

关键机制摘录(sparse_attention_msa.py):

```python
# An E4M3 query attends as is, its scale folded into softmax_scale.
if q_fp8 is None:
    qp, softmax_scale = q[nd:], self.scale
else:
    qp, softmax_scale = q_fp8[nd:], self.scale * q_scale_float
```

## 性能声明

GB300 逐层 `MiniMaxM3SparseMSAImpl.forward`(prefill-only batch,单请求,top-k 16,page 128;3×50 次迭代中位数,波动 ≤1%):

| KV cache | heads | case | main (BF16 q) | 本 PR (E4M3 q) | 降幅 |
| --- | --- | --- | --- | --- | --- |
| FP8 | 16/1 (TP4) | 8k @ 0 | 411 µs | 303 µs | -26% |
| FP8 | 16/1 | 8k @ 32k | 462 µs | 314 µs | -32% |
| FP8 | 16/1 | 16k @ 112k | 801 µs | 530 µs | -34% |
| FP8 | 64/4 (TP1) | 8k @ 0 | 1423 µs | 891 µs | -37% |
| FP8 | 64/4 | 8k @ 32k | 1498 µs | 985 µs | -34% |
| NVFP4 | 16/1 | 8k @ 0 | 379 µs | 342 µs | -10% |
| NVFP4 | 16/1 | 16k @ 112k | 798 µs | 695 µs | -13% |
| NVFP4 | 64/4 | 8k @ 0 | 1411 µs | 1255 µs | -11% |

汇总:FP8 KV 快 26-37%,NVFP4 KV 快 8-13%(相对 main)。仅 bump pin(仍 BF16 q)也部分受益;E4M3 query 在旧 pin 上单独贡献 13-24%。

## 验证与回归风险

GB300(SM103,CUDA 13.4,新 pin 构建)5 个 MSA 测试文件:240 passed,45 skipped(全部为 ROCm-only);MSA 测试文件在 B200 CI 运行。新增 `test_msa_prefill_reads_query_fp8_over_fp8_cache[tp1|tp4]` 断言 `sparse_atten_func` 收到 E4M3 query 且输出匹配 Triton 反量化参考;NVFP4 测试参考改为 E4M3-staged cache 读法(max abs 0.06 vs 0.14;对 BF16 精确反量化有 122/4.1M 元素超出 0.05 容差,故换参考)。旧代码四个 prefill 用例均因 dtype 不符失败。限制:本机验证 SM103,SM100 由 CI 覆盖,SM107 的 E4M3-query NVFP4 内核未测;PR 未含端到端精度评估或 serving benchmark(仅内核级数值与逐层计时)。

## 关联

- MSA 侧:vllm-project/MSA#32(Blackwell prefill port)
- 相邻 MiniMax-M3 MSA PR:#55838、#56932、#59979(仅改 decode 或 indexer metadata,无冲突);Triton decode 回退行为不变的测试见同文件
- 同区域:MiniMax-M3 模型支持与 MSA sparse attention 内核族

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
