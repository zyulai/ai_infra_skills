# PR #30247: Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-07-21 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, run-ci, jit-kernel, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 jit_kernel/dsv4/gemm.py、longcat_flash.py 与两个新增测试文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/30247 |

## 一句话结论

LongCat-Flash 的 bf16 激活 x fp32 路由 GEMM 改用 HPC-Ops `gemm_bf16xfp32` 融合 kernel,router shape 上提速 1.56x–4.31x,LongCat-Flash-Lite e2e 输入吞吐 +2.8%–5.4%。

## 动机

LongCat-Flash 的 router GEMM 是 bf16 激活乘 fp32 路由权重,此前走 `x.float() @ w.t()` 的 fp32 GEMM:既不能利用 bf16 tensor core,又因为激活要升精度到 fp32 而多付一次转换与带宽成本。router GEMM 本身在 forward 中占比不大,但在 prefill(M=token 数很大)时是可观的固定开销。

## 优化点

采用 Tencent HPC-Ops 的 `gemm_bf16xfp32` kernel:fp32 权重一次性分解为两个缓存的 bf16 半份(`w = w_high + w_low/256`,`w_low = ((w - w_high)/scale).bf16`),kernel 把两个 bf16 GEMM 融合执行(split-K + flag workspace),从而在 bf16 tensor core 上保留 fp32 权重的数值精度。dispatch 有多重 guard:仅当 hpc 可用、Hopper(SM90)、shape 满足约束(x/y contiguous、n%64==0、m≥min_m)才走 HPC 路径,否则回退 cublas fp32 GEMM;`LongcatFlashRouter` 按 benchmarked 的 (hidden_size, num_experts) shape 配置 per-shape min-M(Chat 6144x768 → 64,Lite 3072x384 → 128),unbenchmarked shape 与 router_bias 保持原路径;全局可用 `SGLANG_OPT_BF16_FP32_GEMM_ALGO=hpc` 开启。权重 split 与 split-K workspace 按 cache_key(data_ptr/_version/shape/stride/device/dtype)缓存在 weight tensor 上,避免每次前向重复分解。此前 PR 曾内联移植 kernel 源码,本版改为直接消费镜像中构建的 `hpc` 包(由 #31390 引入),后续内核更新只需 bump pinned commit。

## 关键改动

```text
python/sglang/jit_kernel/dsv4/gemm.py: linear_bf16_fp32 新增 hpc 分支,含可用性检测、min-M guard 与权重 split 缓存
python/sglang/srt/models/longcat_flash.py: LongcatFlashRouter 按 shape 的 min-M 表 dispatch 到 linear_bf16_fp32
test/registered/gemm/test_linear_bf16_fp32_hpc.py: 新增,hpc 路径 vs fp32 reference 数值测试
test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py: 新增,router dispatch 逻辑单测(CPU)
```

```python
w_high = y.to(torch.bfloat16)
w_low = ((y - w_high.float()) / _HPC_GEMM_WEIGHT_SCALE).to(torch.bfloat16)
return hpc.gemm_bf16xfp32(x, w_high, w_low, _HPC_GEMM_WEIGHT_SCALE,
                          use_fp32_output=True, use_splitk=True, split_flag=split_flag)
```

## 性能声明

kernel 级(H200,`hpc.gemm_bf16xfp32` vs fp32 GEMM `x.float() @ w.t()`,LongCat router shapes):k=6144,n=768:m=64 → 15.5µs vs 35.6µs(**2.31x**),m=512 → 38.4 vs 120.4(**3.14x**),m=8192 → 430.0 vs 1661.1(**3.86x**);k=3072,n=384:m=64 → **1.56x**,m=512 → **2.13x**,m=8192 → 113.1 vs 487.8(**4.31x**)。

模型级(LongCat-Flash-Lite-FP8,H200 TP1,A/B against origin/main,greedy 输出逐 token 一致;bench_one_batch_server input 1024/output 128):input tok/s bs=1 **+2.8%**(15190→15613),bs=16 **+3.5%**(68588→71021),bs=64 **+5.4%**(62175→65509);output tok/s 变化 -0.4%–+1.0%(bs≤64 低于 Lite 的 min_m=128 guard,decode 有意保持 cublas 路径)。精度:GSM8K 1319 题 parallel 128,main 0.798/invalid 0.000 → 本 PR **0.802**/0.001。

## 验证与回归风险

新增两个注册 CI 测试:H200 上 hpc 路径 vs fp32 reference(rtol 0.08/atol 0.01,upstream tolerance)覆盖两个 router shape、min-M dispatch 与权重 split 缓存复用(kernel 要求 split-K workspace 保持归零,测试断言之);CPU 单测覆盖 dispatch 分支。作者报告 H200 上 "7 passed, 6 subtests passed"。回归风险:仅 sm90a(Hopper)且需环境装 HPC-Ops(#31390 构建进镜像),其他环境全部回退原 cublas 路径、行为不变;rope_theta/rope_scaling 读取方式顺带修改(rope_parameters 优先),与 router 无关但影响 LongCat 配置解析;CI 状态 Base/Extra 均记录失败(:x:),合并前是否修复未注明。

## 关联

前序:#31390(镜像构建 HPC-Ops,使本 PR 直接 import 而非内联移植);同区域:HPC-Ops 后续 H200 tuning 可通过 bump pinned commit 获得。同分类 jit-kernel GEMM 系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
