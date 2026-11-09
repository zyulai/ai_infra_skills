# PR #61903: [Perf][Kernel] Add a CuteDSL BF16x3 router GEMM for fp32 router weights on Hopper

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ischengcheng |
| 合并 | 2026-10-27 |
| 分类 | Attention / Kernel(major,classify 结果,实为 MoE router GEMM) |
| Label | ready, ci/build |
| Diff | 素材未提供 diffstat 数字;pack diff 收录主要文件,3 个文件因超长未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/61903 |

## 一句话结论

为 SM90 补上 CuteDSL BF16x3 router GEMM,替代 fp32 SGEMM 路由算子,H200 上 kernel 提速 1.6–7.7x,MiniMax-M2.5 E2E prefill 吞吐 +4–6%。

## 动机

H100/H200 上 `GateLinear` 对超过 32 tokens 的 fp32 router 权重走 `F.linear(x.float(), w)`,即 CUDA core 上的 fp32 SGEMM。SM100 已有 CuteDSL BF16x3 kernel(#47973、#55899),gfx950 有 hipBLASLt 版(#52668),SM90 一直空缺。受影响模型众多:MiniMax-M2/M2.5/M3、GLM-4.5/4.6/4.7 与 GLM-4.5-Air、Hunyuan-V3/V4、Step-3.5-Flash、ERNIE-4.5、Ling 2.0(均 `router_dtype: fp32`)等,MoE 每层的 router GEMM 都在为这条慢路径付钱。

## 优化点

CuteDSL kernel 的关键机制:MMA 维度交换 —— 64 个 experts 构成 wgmma 的 M 维、tokens(每 CTA 32–128)作 N 维,小 batch 也能填满指令;一个 TMA warp 搬 fp32 weight tiles 与 bf16 token tiles 进 shared memory;MMA warpgroup 在寄存器里把 fp32 权重的每个 k16 slice 拆成 3 个 bf16 项(每项保留剩余值的最高 16 位,三个 8-bit 尾数段覆盖 fp32 的 24 位,三项之和 bit-exact 还原权重),与 bf16 激活做 register-A wgmma。数值上:首项乘积走每个 K-tile 清零的 tensor-core partial 再累入 fp32 寄存器累加器,两个约 2^-8 量级的小项跨 K 保留各自 partial(与 SM100 kernel 一致);Split-K 用最多 8 CTA 的 thread-block cluster 经 DSMEM 按 rank 顺序归约,确定性、无 workspace、无第二 kernel,无 host sync,可安全 capture 进 CUDA graph;PDL 常开。不做 autotune,`_pick_config` 用 H200 拟合的固定成本模型从 4 tiles × 4 splits 中选配置;fp32 GEMV 的 SM90 cutoff 从 32 降到 24 tokens(交叉点实测 18–28 tokens)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/router/bf16x3_router_gemm_sm90.py: 新增 SM90 CuteDSL BF16x3 router GEMM(自定义 op torch.ops.vllm.sm90_bf16x3_router_gemm)
vllm/model_executor/layers/fused_moe/router/gate_linear.py: tier 4 开放给 Hopper;fp32 GEMV 上限 24 tokens;注册 warmup
vllm/model_executor/layers/fused_moe/router/bf16x3_router_gemm_cutedsl.py: 更名 bf16x3_router_gemm_sm100.py
tests/kernels/test_bf16x3_router_gemm_cutedsl.py: 新增 9 个真实 router 形状的 SM90 测试
.buildkite/test_areas/kernels.yaml: 新增 "(H100) BF16x3 Router GEMM Kernels" CI step
```

关键片段(寄存器内 fp32→3×bf16 拆分,PTX 内联):

```
and.b32 m0, u0, 0xffff0000;   // 首项:高 16 位
sub.f32 r0, u0, m0;           // 余数
and.b32 n0, r0, 0xffff0000;   // 次项
sub.f32 l0, r0, n0;           // 三项之和 bit-exact 还原 w
```

## 性能声明

- Kernel(1×H200,CUDA graph 内、cold L2、10 个 router 形状):33–512 tokens 提速 1.6–3.8x,1024–16384 tokens 提速 3.3–7.7x。例:GLM-4.5 (5120,160) M=16384:985.5→157.0 µs(6.28x);ERNIE-4.5 (8192,64) M=16384:721.2→93.2 µs(7.74x);MiniMax-M2 (3072,256) M=64:15.9→6.0 µs(2.66x)。
- 精度:对 fp64 参考的 MAE 为被替换 fp32 SGEMM 的 0.16–0.73x(所有形状与 token 数);10 个形状均无 top-8 选择差异(fp32 SGEMM 在 2 个形状上各有 1/16384 行差异)。
- E2E(MiniMax-M2.5 FP8,TP2,2×H200,同容器交替 main/PR 各 4 轮):8K/32K prompt median TTFT −4.6%/−3.2%;8K 并发 16 吞吐 3.56→3.77 req/s(+5.8%);混合 1K-in/128-out 并发 64 输出 tok/s +4.6%、TPOT −4.0%;GSM8K 准确率均值 0.9327→0.9339、wall time −2.5%。
- Beam-1.0 FP8 单 TP8 rank 代理(dummy weights):8K prefill chunk 155.3→133.5 ms(−14.0%);decode 64/128 请求 −7.4%/−10.9%。
- Eager 模式单次调用 host 时间 17–18 µs,main 为 21–36 µs。

## 验证与回归风险

SM90 单测:9 个真实 router 形状 × 7 个 token 数(8–8192)断言 MAE 不高于 fp32 SGEMM;partial K/expert tile;one-hot 激活下权重拆分 bit-exact;GateLinear dispatch 边界(24/25 tokens 等);`torch.compile fullgraph=True` 可追踪。CI 新增 H100 step。风险面:成本模型在 H200(132 SM)拟合,其他 SM90 器件选择可能非最优;单 token 时 3/10 形状 fp32 GEMV 更准(保持在 GEMV 路径);guard-free torch.compile 下 tier-3 分支可能在首个 trace 尺寸被固定(issue #60681);Beam-1.0 数字来自 dummy weights 的单 rank 代理,需 #62074 合并后做完整 8×H200 验证。

## 关联

同族 kernel:#47973/#55899(SM100 BF16x3,本 PR 的蓝本)、#52668(gfx950 hipBLASLt)、#34302(DSV3 TRTLLM router GEMM)。相关 dispatch 规则:#62042、#62126、#61727;替代方案 #49312(hpc-ops 外部库,opt-in)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
