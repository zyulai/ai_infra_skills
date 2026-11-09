# PR #64481: [Perf][GLM-5.3] Enable GLM low-latency BF16 GEMM on SM100

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @namgyu-yun |
| 合并 | 2026-10-29 |
| 分类 | Attention / Kernel(major) |
| Label | ready, deepseek, glm |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含内核表文件与对应测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/64481 |

## 一句话结论

为 GLM-5.2/5.3 低延迟 BF16 decode GEMM 增加 B200 (SM100) 实测 dispatch 表替代 cuBLAS,各投影在 M≤8 提速 1.14–2.35x。

## 动机

GLM 低延迟 BF16 decode GEMM 此前已有 SM103(B300)与 SM107(VRNVL72)实测表,SM100(B200)直接走 cuBLAS。 skinny BF16 GEMM(M≤8 的 decode 投影)在 cuBLAS 下有明显优化空间,需要按 (N,K) 与 M 实测的 plan 表接管。

## 优化点

在 glm52_low_latency_gemm.py 新增 GLM52_PROJECTIONS_SM100 表,覆盖 8 类投影:q_a+kv_a、q_b TP8、q_b TP4/indexer wq_b、o_proj TP8、indexer wk+weights、shared gate_up TP8/TP4、MTP eh_proj;每个投影按 M=1–8(部分只到 M=4)配置 SkinnyGemmConfig(cute config,含 block/warp/unroll/vector_width/static_k 等)。_low_latency_table() 增加 device capability (10,0) 分支返回新表。入表门槛:比 cuBLAS 快 ≥5%(CUPTI、CUDA graph、cold 与 hot L2 双条件),不满足的 cell 留在 cuBLAS;o_proj TP4 与 shared expert down 投影仍走 cuBLAS。

## 关键改动

```text
vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py: 新增 _SM100_* 投影
  定义与 GLM52_PROJECTIONS_SM100,(10,0) 分发
tests/kernels/test_bf16_skinny_gemm.py: GLM_GPU_TABLES 与 plan 断言加入 (10,0) 表
```

## 性能声明

B200,cold L2,各投影 M=1 cuBLAS→表(µs)与对 routed M 的加速比:q_a+kv_a (2624,6144) 12.00→8.48,1.21–1.42x;q_b TP8 (2048,2048) 6.02→4.06,1.46–2.00x;q_b TP4/indexer wq_b (4096,2048) 6.94→5.76,1.15–1.29x;o_proj TP8 8.03→6.91,1.14–1.16x;indexer wk+weights (160,6144) 6.08→2.62,1.48–2.35x;shared gate_up TP8 8.00→4.26,1.37–1.88x;TP4 9.09→5.22,1.16–1.74x;MTP eh_proj 37.63→26.24,1.31–1.43x。总体 1.14–2.35x over cuBLAS per projection at M ≤ 8。无 e2e 数据(需 4–8 张 B200)。

## 验证与回归风险

pytest tests/kernels/test_bf16_skinny_gemm.py -k "glm or dsv3" → 84 passed, 180 skipped(B200);run_glm52_plan vs F.linear 全部 40 个 routed cell 在 cold/hot L2 下 ≥1.05x。风险限于 GLM/DSv3 系列 decode 投影的 GEMM 分发:仅当 (N,K) 命中表且 M 在测量范围内才替换 cuBLAS,其余路径不变。

## 关联

#59822(同区域相关 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
