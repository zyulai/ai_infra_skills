# PR #52606: [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kliuae |
| 合并 | 2026-08-21 |
| 分类 | Attention / Kernel |
| Label | rocm, ready, ci/build, cpu, kimi, k3 |
| Diff | diffstat 缺失;pack 明示 diff 因超长截断(9 个文件未含) |
| 链接 | https://github.com/vllm-project/vllm/pull/52606 |

## 一句话结论

Kimi-K3 KDA prefill 的 9 个 Triton kernel 融合为两 kernel HIP 链,chunk 阶段快至 3.5x、TTFT −4%。

## 动机

ROCm 上 Kimi-K3 的 KDA prefill chunk 路径是一条九连发 Triton kernel 流水线,launch 与中间物落 HBM 开销大;其中 chunk recurrence 对每条序列是串行状态更新,按序列数派发 workgroup 时 CU 占用很低,长 prefill 下尤其浪费。

## 优化点

新增 `csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu`,用两个 HIP kernel 替代 Triton dispatch 链:`kda_chunk_prologue` 每个 workgroup 处理一个 head 的 64-token chunk,一次产出 recurrence 所需全部量(L2 归一化并 gated 的 q/k、intra-chunk 注意力矩阵 Aqk、transition 逆矩阵 Akk、w/u),chunk 间可并行故按 `(chunk, head)` 网格派发,三角形求逆保持 16x16 对角块 fp32 精度、宽的部分用 MFMA 走 GEMM;`kda_chunk_fused` 沿 (sequence, head) 的 chunk 前进并把 [V, K] state 常驻 MFMA 累加器,per-chunk state 与重算 value 均不写 HBM。针对序列内串行,把一条序列的 chunk 切成 G 组(组数按 device capability 定)做两遍法:pass 1 各组从零 state 走并记录组转移算子,`kda_group_scan` 在组间做 scan 精确恢复各组真实初始 state,pass 2 从正确起点重放产出输出。对外封装为两个 PyTorch 接口替代原 Triton 链,新增 engine 参数 `--kda-prefill-backend=fused` 并在支持时默认启用;build/dispatch 目前 guard 在 gfx950。

## 关键改动

```text
csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu: kda_chunk_prologue / kda_chunk_fused / kda_group_scan
(其余 9 个文件被 pack 截断,含 Python 封装与 --kda-prefill-backend 参数)
```

```text
Per chunk, with S the state at the chunk boundary:
  v_new = u - w S^T
  o     = scale (q exp2(g)) S^T + tril(Aqk) v_new
  S     = S diag(exp2(g_last)) + v_new^T kg
```

## 性能声明

- Kernel microbench(tokens/seqs,Triton chain → fused chain):1024/1 为 139.2→79.1 µs(1.76x);4096/1 为 330.5→143.0 µs(2.31x);4096/8 为 256.6→145.2 µs(1.77x);16384/1 为 1232.4→393.5 µs(3.13x)。chunk stage 单独:4096/1 为 301.1→113.7 µs(2.65x),16384/1 为 1174.2→335.9 µs(**3.50x**)。总结:融合链约 **2x faster**,chunk recurrence 阶段尤甚。
- E2E(`vllm bench serve`,Kimi-K3,random 8192/1024,TP8):conc=16/32/64/128 下 total tokens/s +1.1%/+1.3%/+2.3%/+3.0%,mean TTFT −4.4%/−4.4%/−4.4%/−4.3%(声明 **~4%**),TPOT −0.9%~−2.8%。
- GSM8K 20-shot(MI355X):flexible-extract 0.9765±0.0042,strict-match 0.9773±0.0041。

## 验证与回归风险

测试计划为融合 kernel 单元测试 + MI355X 上的 lm_eval GSM8K(结果如上)。风险:kernel 目前只在 gfx950 测试,build/dispatch 亦 guard 于 gfx950;支持时默认启用意味着 ROCm Kimi-K3 prefill 默认切换到新路径,数值行为依赖组扫描的精确重组(scan 中 decay-only 部分保持 fp32、修正项过 bf16 MFMA);pack 明示 diff 截断(9 个文件未含),Python 封装与测试细节需看 PR。

## 关联

同批 Kimi-K3/KDA 方向:#53196(FlashKDA checkpoint 单次打包导出)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
