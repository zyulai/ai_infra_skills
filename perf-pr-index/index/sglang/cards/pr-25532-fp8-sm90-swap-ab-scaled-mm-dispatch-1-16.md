# PR #25532: [fp8] SM90 swap-AB scaled_mm dispatch (~1.16x kernel geomean, +5.8-18.5% end-to-end)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-05-20 |
| 分类 | MoE(major) |
| Label | sgl-kernel, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数;Files changed 列表见 PR 正文) |
| 链接 | https://github.com/sgl-project/sglang/pull/25532 |

## 一句话结论

SM90 FP8 W8A8 GEMM 小 M 路径改用 swap-AB 分桶调度,H200 kernel geomean 1.156x,Llama-3.1 FP8 端到端 decode 吞吐 +5.8-18.5%。

## 动机

main 的 SM90 FP8 dispatch 对所有小 M 路径统一用 tile `<_64, _64, _128>`:decode batch=1..16 时 M_orig 远小于 64,M-tile 大部分是 padding 计算(M=1 时仅 1/64 是真实工作)。swap-AB 在 kernel 层转置 GEMM(D^T = B^T · A^T),让 kernel M 对应大得多的 N_orig、kernel N 对应小的 M_orig,再配 16/32 的小 kernel-N tile,使 tile 恰好略高于 M_orig,把 padding 浪费压到最低——这是小 batch decode 的主导开销。该路径覆盖所有 FP8-W8A8 量化模型的 dense linear(Llama/Qwen/Mistral/Mixtral 等 `*-FP8-dynamic`/per-token+per-channel checkpoint,经 `apply_fp8_linear → fp8_scaled_mm`);DeepSeek V3/R1 的 blockwise FP8 走独立 kernel,不受影响。

## 优化点

把 `fp8_gemm_kernel.cu` 内联的 SM90 dispatch 替换为 `cutlass_extensions/` 下的模块化 swap-AB 分桶调度:按 (M, N) 分 8 个 bucket,每个指定 tile/cluster/是否 swap;其中两个新 bucket 是为封堵朴素 swap-AB 遗留的回退区——M32_N8192(kernel-N=32 恰好放下 M=32,把 M=32 从相对 main -9% 拉到 +19%)与 M128_smallN(N≤4096 时回退小 tile,修复 M128 大 tile 在该区域 20-25% 的回退)。新增 SM90 col/row broadcast epilogue(运行时选择 per-tensor 或 per-channel scale,one kernel 覆盖所有量化组合)、bias/azp EVT 树,以及 array 变体(grouped GEMM 用)。dispatch 切换点只改 6 行,并删除 355 行不再可达的旧 dispatch 代码。开发中还尝试并回退了 Stream-K(与 TileM=64 结构冲突)、FastAccum(M>128 上 -8%)、max_swizzle_size(-50%)。

## 关键改动

```text
sgl-kernel/csrc/cutlass_extensions/gemm/fp8_gemm_sm90_dispatch.cuh: 新增;8 桶 swap-AB 分桶调度(~480 行)
sgl-kernel/csrc/cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp: 新增;row/col-or-scalar broadcast(SM90 EVT)
sgl-kernel/csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp: 新增;ScaledEpilogue[Bias|ColumnBias|BiasAzp*] EVT 树
sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu: dispatch 切到 cutlass_scaled_mm_sm90_fp8,删 355 行死代码
sgl-kernel/tests/test_fp8_gemm.py / benchmark/bench_fp8_gemm_swap_ab.py: 60 用例 + 定向 benchmark
```

```text
# dispatch 桶(tile / cluster / swap)
M16_N1280   M∈[1,16]   N≤1280  <64,16,256>  <1,2,1>  swap
M16_N8192   M∈[1,16]   N>1280  <64,16,256>  <1,1,1>  swap
M32_N8192   M∈(16,32]  N>1280  <64,32,256>  <1,1,1>  swap   # 新增,封堵 M=32 回退
M64_N1280   M∈(16,64]  N≤1280  <64,16,256>  <1,4,1>  swap
M64_N8192   M∈(32,64]  N>1280  <64,64,256>  <1,1,1>  swap
M128_smallN M∈(64,128] N≤4096  <64,64,128>  <1,1,1>  -      # 新增,封堵 M128 回退
M128        M∈(64,128] N>4096  <64,128,128> <2,1,1>  -
default     M>128      *       <128,128,128><2,1,1>  -
```

## 性能声明

Kernel 级:H200(SM90),fp8_e4m3 x bf16,per-token A scale + per-channel B scale,无 bias,`do_bench_cudagraph(rep=200, median)`,M≤128 共 84 形状:geomean 1.156x(58 胜 4 负),各 M 桶均 ≥1.05x(M=1 1.13x、M=8 1.22x、M=16 1.24x、M=32 1.19x、M=64 1.09x、M=96 1.19x、M=128 1.05x);最大单形状 M16/N8192/K4096 2.02x;4 个形状回退均 ≤10%(如 M1/4096/4096 0.97x,main 在该 sweet spot 有 cluster multicast 优势)。M>128 路径不变(geomean 1.000x ± noise)。

端到端:H200,`sglang.bench_one_batch`,input_len 32 / output_len 64,TP1,Llama-3.1-70B-Instruct-FP8-dynamic:bs1 42.31 → 49.03 tok/s(+15.9%,23.64 → 20.40 ms/tok)、bs8 +18.5%、bs16 +15.3%、bs32 +13.9%;Llama-3.1-8B/70B FP8-dynamic 合计 +5.8-18.5%。70B 收益是 8B 的 2.7-5 倍(hidden 8192、80 层、更受 HBM 带宽约束)。

## 验证与回归风险

`test_fp8_gemm.py::test_accuracy_sm90_swap_ab` 新增 60 用例(15 形状 x bf16/fp16 x bias/no-bias,覆盖每个 bucket 边界)全过;既有 500 用例 `test_accuracy` 仍过。风险:swap-AB 在结构上无法使用 main 在 M=1 sweet spot 的 `<_1,_8,_1>` cluster A-multicast 方向,个别形状有 ≤10% 回退(已列明);M>128 未动;测试清单中"CI 通过"一项未勾选(正文 CI States 显示 Base CI 绿)。

## 关联

epilogue 与 dispatch 改编自 vllm-project/vllm 同名文件(`scaled_mm_sm90_fp8_dispatch.cuh`、`broadcast_load_epilogue_c3x.hpp`,commit 16bff144)。pack 未列出其他前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
