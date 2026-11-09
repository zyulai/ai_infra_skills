# PR #65791: [Perf] Vectorize silu_and_mul_per_block_quant for large inputs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-11-06 |
| 分类 | Quantization (tier: major) |
| Label | quantization |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/65791 |

## 一句话结论

为 fused silu_and_mul+block quant 增加向量化变体,大输入(prefill)下 kernel 快 2.0-2.45x,Qwen3-30B-A3B-FP8 prefill 端到端 +5.6%。

## 动机

`silu_and_mul_per_block_quant`(fused SiluMul+BlockQuant)原实现每个 (token, group) 启动一个 128 线程 block,每线程只发一次 2 字节 load,group max 靠 shared-memory 树归约。prefill 尺寸下该 kernel 在 H200 上只跑到约 0.9 TB/s,反而比它所替代的 unfused `silu_and_mul` + `per_token_group_quant_fp8` 组合慢 1.3-2.2x——融合本应赢却在大输入上输了,这是本 PR 要解决的矛盾。

## 优化点

新增向量化变体 kernel:每个 group 由 `group_size / 8` 个 lane 处理(group 128 时 16 lane),每 lane 用 16B(`uint4`)对齐 load 读 gate/up 各 8 个连续 16-bit 元素,group max 用 `__shfl_xor_sync` warp shuffle 归约(无 shared memory、无 `__syncthreads`),lane 0 写 scale,量化输出以 8B(`uint2`)store。算术与原 kernel 完全一致(`gate * (1/(1+exp(-gate))) * up`、`max(amax/qmax, min_scale)`),结果 bitwise 相同,文件不启用 `--use_fast_math`。调度按实测阈值分发:`num_tokens * num_groups >= 3072` 且输入 16-bit、输入 16B 对齐、输出 8B 对齐时走向量化版;≤2048 groups 时 one-block-per-group kernel 反而快约 0.4-0.9 us,故保留原 kernel 服务小输入(decode 延迟不变);非对齐视图自动回落旧 kernel。

## 关键改动

```text
csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu: 新增 silu_and_mul_per_block_quant_vec_kernel(warp shuffle 归约 + 16B/8B 向量访存)与 use_vec 分发逻辑
tests/kernels/core/test_fused_silu_mul_block_quant.py: 新增 vec vs scalar bitwise 等价测试
```

## 性能声明

Kernel 微基准(H200,bf16 in / fp8 out,group 128,CUDA-graph 计时,p10 us):rows=16384、2I=512 时 23.88→12.04 us(1.98x),2I=4096 时 197.16→82.37 us(2.39x),2I=24576 时 1166.89→478.78 us(2.44x);rows=65536、2I=4096 为 783.51→320.24 us(2.45x);rows≤64 时与旧 kernel 持平(1.00x 左右)。对比 unfused 组合:≥16k rows 时快 8-53%(如 2I=512/rows=16384:fused 本 PR 12.04 vs unfused 18.77)。端到端(Qwen/Qwen3-30B-A3B-FP8,H200 TP1,`--moe-backend auto`(Triton),`VLLM_USE_DEEP_GEMM_E8M0=0` 即 main 已走 fused kernel 的配置,每 arm 两次运行交错执行):prefill tok/s 65,752/64,061 → 67,634/69,375(+5.6% 均值),mixed +1.7%(7,630/7,781 → 7,855/7,829),decode TPOT 不变(c1 3.92/3.90,c8 5.86/5.85→5.89/5.87,c32 8.33/8.54→8.45/8.24 ms)。workload:`vllm bench serve --dataset-name random --ignore-eos`,prefill 4096 in/1 out conc 64、mixed 1024/512 conc 128、decode 1024/256 at c1/8/32。

## 验证与回归风险

测试:新增 `test_silu_and_mul_per_block_quant_vec_matches_scalar`——用同一输入的 2 字节错位视图强制走 scalar 路径,断言 bf16/fp16 x fp8/int8 x group 64/128 x 两种 scale 布局下 bitwise 相等;既有参考测试经大形状(512x5120、256x14336、int32 offset overflow)已覆盖向量化路径;`tests/kernels/core/test_fused_silu_mul_block_quant.py` 347 passed(main 331 + 新 16),compile fusion 测试 126 passed 与 main 持平。风险面:仅 16-bit 输入 + 满足对齐与阈值时才切换,decode/小输入路径不变;对齐判断基于指针模 16/8,错位视图自动回落;与 #62386(TMA scale stride,同文件)需 rebase,与 #43399(UE8M0)需合并 template flags。

## 关联

同区域:#65794(前置于它——其把 fused 路径设为默认,替代 #64906 与 #43399;旧 kernel 下该改动会使 Qwen3-30B-A3B-FP8 prefill -3.8%,本 PR 消除该回退)、#62386(同文件 TMA 对齐 scale stride)、#43399(UE8M0 选项,同一 kernel)。受益路径:Triton FP8 MoE(`VLLM_USE_DEEP_GEMM_E8M0=0`)与 SiluMul+BlockQuant compile fusion。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
