# PR #40996: [AMD] Add tuned dsv4 shape

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @akao-amd |
| 合并 | 2026-09-24 |
| 分类 | Attention / Kernel | tier: major |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 为 8 个新增 JSON 调优配置文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/40996 |

## 一句话结论

为 gfx1250/MI45x 添加 DSV4-Flash 全部 2D fp8 权重形状的 GEMM autotune 配置,DSV4-Flash serving 吞吐 +5.86%、延迟 -4.87%。

## 动机

`_w8a8_block_fp8_matmul_gfx1250` 在 AMD Instinct Graphics(gfx1250/MI45x)上缺少针对 DSV4-Flash 2-D fp8 权重形状的 autotuned JSON 配置。没有这些配置时,runtime 会回退到固定的 BLOCK_SIZE_M=64——在 decode 的 token=1 形状下,一个 tile 的 64 行里浪费了 63 行,M 维度并行度几乎为零,decode GEMM 显著低于硬件应有水平。这是典型的"kernel 已有、配置缺失"型性能缺口,补配置即可解锁,不需要动 kernel 本身。

## 优化点

纯配置文件改动:新增 8 个 JSON 调优配置,路径形如 `configs/N=<N>,K=<K>,device_name=AMD_Instinct_MI455X,dtype=fp8_w8a8,block_shape=[128, 128].json`,覆盖 (N,K) 组合 512/1024/2048/4096/8192 x 1024/2048/4096/8192(DSV4-Flash 的 2-D fp8 权重形状)。每个文件按 batch/token 维度 M 从 1 到 2048 逐档给出 BLOCK_SIZE_M/BLOCK_SIZE_N/BLOCK_SIZE_K、GROUP_SIZE_M、num_warps、num_stages 与 waves_per_eu(AMD 特有的每 EU wave 数调节)。关键模式是 M=1-16 的小档全部选 BLOCK_SIZE_M=16(部分配 num_warps=2、waves_per_eu=4),直接消除 BLOCK_SIZE_M=64 fallback 在 token=1 时 63/64 行的浪费;大 M 档(512-2048)则转向 BLOCK_SIZE_M 64-256、BLOCK_SIZE_N 64-256 的吞吐取向,并按形状把 BLOCK_SIZE_K 降到 32-64。kernel 代码零改动,查表 miss 时仍走原有 fallback。

## 关键改动

```text
python/sglang/kernels/ops/quantization/configs/N=1024,K=4096,...MI455X...json: 新增,按 M 1-2048 调优
python/sglang/kernels/ops/quantization/configs/N=2048,K=4096,...MI455X...json: 新增,仅 M 1-16 档
python/sglang/kernels/ops/quantization/configs/N=32768,K=1024,...MI455X...json: 新增(其余 N=4096,K=2048/4096/8192、N=512,K=4096、N=8192,K=4096 同为新增)
```

```text
M=1 档(N=1024,K=4096): BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128,
GROUP_SIZE_M=8, num_warps=2, num_stages=1, waves_per_eu=4
```

## 性能声明

条件:DSV4-Flash,ISL=8K / OSL=1K / Concurrency=4~64 benchmark sweep(硬件档位为 AMD Instinct MI455X/gfx1250 配置文件;PR 未注明 GPU 数与其他 server 参数):

- Throughput(output tok/s)整体 improved 5.86%:conc 4:132.0→142.5(+8.0%);conc 8:231.2→250.6(+8.4%);conc 16:380.7→405.9(+6.6%);conc 32:598.4→622.4(+4.0%);conc 64:922.2→958.5(+3.9%)。
- Latency 整体 reduced 4.87%:TPOT mean -6.9%~-2.8%(conc4 26.0→24.2 ms);ITL mean -7.0%~-3.4%;E2E mean -6.8%~-3.3%(conc4 12,378→11,531 ms);TTFT mean 反升 +1.1%~+2.9%(conc4 237.7→244.5 ms)。

## 验证与回归风险

准确性:正文仅写 "Intact." 并附一张截图(未给出可读数字);checklist 各项均未勾选,PR 未列具体单测。回归风险:影响面限于 gfx1250/MI45x 上走 `_w8a8_block_fp8_matmul_gfx1250` 且命中这 8 个 (N,K) 形状的调用(主要是 DSV4-Flash);配置按 (N,K,M) 查表,未命中形状不受影响;TTFT 有 1-3% 的反升(高并发下 prefill 侧 GEMM 档位选择),属于已知取舍;其他硬件与其他模型路径零改动。

## 关联

无(正文未列同区域其他 PR;机制上同属 AMD block-fp8 GEMM tuned-config 系列)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
