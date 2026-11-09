# PR #50654: [ROCm][Perf] Kimi-K3 Fused kernel for KDA decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kliuae |
| 合并 | 2026-08-12 |
| 分类 | Attention / Kernel |
| Label | performance, rocm, ready, ci/build, kimi, k3 |
| Diff | diffstat 缺失(pack 未提供,正文 diff 另有 5 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/50654 |

## 一句话结论

为 ROCm 新增 Kimi-K3 KDA 单 token decode 融合内核,把三连发 Triton kernel 合成一个,内核延迟最高 2.31x,e2e 输出吞吐 +~7%。

## 动机

Kimi-K3 的 KDA 层在每个 decode step 对每条序列、每个 value head 做三件事:Q/K/V 投影上的 causal conv1d 更新(推进每序列卷积状态)、gated delta-rule recurrence(更新 [head_dim, head_dim] 循环状态并产出 attention 输出)、输出上的 gated RMSNorm。ROCm 路径此前由三个独立 Triton kernel(另加两次中间拷贝)串行完成,且在 69 个 KDA 层中每层都重复;本 PR 对齐 CUDA 路径,补上 HIP 版融合内核。

## 优化点

新内核 `fused_kda_decode` 把 conv1d 更新、delta-rule recurrence、gated RMSNorm 融合为一次 launch。recurrent state 每 (序列, head) 为 [128,128] fp32,每 token 读写一次,是 128 KiB HBM 流量,内核以此为带宽瓶颈设计:prologue 先整体非临时预取(`__builtin_nontemporal_load`)整个 state 切片,让其飞行于 conv1d、RMS 归约与两个 barrier 之下;state 读写均用 non-temporal 访存;归约用 wavefront 内 DPP 行归约加一次 LDS 跨组归约;一次三路归约同时得到两个 RMSNorm 与折叠进输出的 rank-1 state 更新(输出 `o = h'.q + vn * (k.q)`)。对 CUDA-graph decode batch 尾部的 `NULL_BLOCK_ID`(slot <= 0)padded 行直接置零返回,避免无谓读写 slot 0 状态。仅支持单 token decode step,build 与 dispatch 目前在 gfx950 上 guard。

## 关键改动

```text
csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu: 新增 800 行 HIP 融合内核(conv1d + delta-rule + gated RMSNorm)
benchmarks/kernels/benchmark_kimi_k3_kda_decode.py: 内核 microbenchmark,支持 eager 与 CUDA-graph replay 两种计时
```

其余 Python 侧接线(算子注册、模型调用)在 pack 中被裁剪的 5 个文件内。模板 dispatch 按 heads(12/24/48/96)、conv state 更新、lower_bound、beta sigmoid 等开关实例化。

## 性能声明

内核延迟(rows 为 token 数,Triton chain 为基线):1 行 10.98 µs → 5.76 µs(1.91x);8 行 14.21 → 6.15 µs(2.31x);32 行 21.98 → 12.15 µs(1.81x);64 行 31.94 → 19.99 µs(1.60x);128 行 51.14 → 37.72 µs(1.36x)。e2e(MI355X,Kimi-K3,ISL/OSL 8192/1024,concurrency 64):total tokens/s 7036.48 → 7513.44(+6.78%),output tokens/s 781.83 → 834.83(+6.78%),mean TPOT 75.26 → 70.15 ms(-6.79%),mean TTFT 基本持平(-0.43%)。

## 验证与回归风险

质量侧用 lm_eval gsm8k(MI355,5-shot/20-shot):PR 与 baseline 的 exact_match 基本一致(5-shot flexible 0.9689 vs 0.9689,20-shot flexible 0.9689 vs 0.9697)。新增内核 microbenchmark 覆盖 eager 与 graph 两种模式。风险:内核仅测过 gfx950,build/dispatch 被 guard 在 gfx950;仅覆盖单 token decode step(prefill/多 token 路径仍走 Triton);heads 数硬编码支持 12/24/48/96;依赖 SD cache layout 的 conv_state/state stride 约束。

## 关联

无(本批索引中无同区域 KDA/ROCm 前后序 PR 编号;PR 正文提及 baseline 含 PR #50649 的 fixes)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
