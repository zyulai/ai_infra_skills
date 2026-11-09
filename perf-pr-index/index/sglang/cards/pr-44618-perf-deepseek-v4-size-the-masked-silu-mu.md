# PR #44618: [Perf] deepseek_v4: size the masked silu-mul-quant grid to the GPU, not the padded capacity

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Monishver11 |
| 合并 | 2026-10-19 |
| 分类 | MoE (tier: major) |
| Label | quant, run-ci, jit-kernel |
| Diff | pack 未提供 diffstat（diff 内容可用：2 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/44618 |

## 一句话结论

DeepSeek-V4 masked SiLU-mul-quant kernel 把 grid 从 T*topk 个 CTA 缩到一张 resident wave，64 tokens/GPU 时 17.1 -> 3.3 us，输出位级一致。

## 动机

该 kernel 跑在 DeepEP low-latency decode 的每个 MoE 层（DSV4 因 `swiglu_limit` 必走此路径），但 host 无法读 GPU 上的 `masked_m`（会 sync），旧实现按 padded capacity 启动 `T * topk` 个 CTA、每个 CTA 先做一次 `masked_m` 前缀扫描再判断是否有真行。decode 时真行极少：Flash TP4 形状启动 24,576 个 CTA 处理约 400 行，H100 上即使 0 真行 kernel 也要 15.9 us——几乎全是空 CTA 过调度器。

## 优化点

改为 persistent kernel：grid 固定为 `min(SMs x resident CTAs, T * topk)`（H100 上 1,056 个），occupancy 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 按 block size 缓存。每个 CTA 只对 `masked_m` 做一次前缀扫描存入共享内存（1KB），无行可处理的 CTA 在 PDL wait 前直接退出；然后各 CTA 以 grid stride 认领扁平化真行列表的行，行所属 expert 由对 `s_row_end` 的二分查找得出。逐行计算体原样搬进 `silu_mul_quant_varlen_row`，数学与输出字节不变；`max_rows = T * topk` 保留旧隐式上限。同一思路此前已见于 vLLM 的 Silu v2 `persistent_masked_m_silu_mul_quant`（vllm#25074）。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh: persistent grid + scan_masked_m/find_expert + 逐行函数抽取 + occupancy 缓存
test/registered/kernels/ops/quantization/test_per_token_group_quant.py: 新增 test_masked_post_quant_writes_exactly_the_real_rows（8 个 case）
```

## 性能声明

H100 80GB，`do_bench_cudagraph`，DSV4 cookbook DeepEP 形状、group 128、swiglu_limit 10、均匀 top-6 路由。Flash EP4 T=4096（B200/B300/GB200 TP4 形状）：0 行 15.92->1.63 us（9.74x）、64 tok/GPU 17.10->3.26 us（5.25x）、dispatch cap 1024 tok 49.89->45.89 us（1.09x）；Flash EP8 T=8192：64 tok 32.19->3.32 us（9.70x）、0 行 16.25x；Pro EP4 T=1024（GB300 TP4）：64 tok 8.64->4.32 us（2.00x）；Flash EP4 T=1024（H200 FP8 TP4）64 tok 5.84->3.17 us（1.84x）；Pro EP16 T=2048（H200 FP8 TP16）64 tok 14.33->4.29 us（3.34x）。所有 decode 点至 dispatch cap 快 1.09x-16.25x；唯一变慢的 "all real" 场景（每行都真）慢 2-4%。作者按 microbench 推算 Flash B200 TP4 64 tok 时每 decode step 约 0.6 ms 节省（43 层 MoE），注明为推算、非 e2e 实测。

## 验证与回归风险

新增测试验证真行与 contig kernel 逐字节相等、padded 行 sentinel 未被写，覆盖 all-zero/one-hot/uneven/full 四种 `masked_m` 与两种 scale layout；该测试能抓出三种人为注入的映射错误。作者报告 464/464 位级一致（含 5 shapes、group 32/128、3 种 scale layout、swizzle 开关、swiglu_limit 开关）；`test_per_token_group_quant.py` 121 passed（8 新增），另一调用方 `test_humming_fp8_dispatch.py` 2 passed；寄存器 32/thread 不变、0 spill。风险："all real" 极端场景慢 2-4%（作者保留了 stride loop 方案）；B200 的 transposed UE8M0 layout 未实测（无硬件）；与 #32066 在同 launcher 行冲突需 rebase。

## 关联

同思路前序：vLLM vllm-project/vllm#25074（Silu v2 persistent masked kernel）；文件冲突 PR：sgl-project/sglang#32066（同 launcher 行加 launch guard）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
