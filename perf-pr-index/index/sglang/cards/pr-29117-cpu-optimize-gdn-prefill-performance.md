# PR #29117: [CPU] optimize GDN prefill performance

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mingfeima |
| 合并 | 2026-06-25 |
| 分类 | Other(tier: major) |
| Label | sgl-kernel, intel, cpu, run-ci |
| Diff | pack diffstat 缺失;diff 为 sgl-kernel/csrc/cpu/mamba/fla.cpp 大规模重写(pack 另注明 3 个文件因超长从 diff 中省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/29117 |

## 一句话结论

重写 CPU GDN(Gated Delta Network)fla.cpp kernel:融合 intra/inter-chunk 阶段并用 AVX512/AMX 指令,对比现有实现约 1.5x。

## 动机

优化 fla.cpp 中的 CPU GDN kernel,使其更好地匹配 fused Triton pipeline(GPU 侧结构),提升 AVX512/AMX CPU 上的 prefill/decode 性能,目标是对现有实现约 1.5x 加速。PR 正文未展开改动前的 profile 细节。

## 优化点

核心是消除中间物化与向量化(据 diff 与代码内注释):intra-chunk 阶段融合 `kkt_solve`、下三角求解与 `recompute_w_u`,不再物化 A 矩阵;inter-chunk 阶段融合 `recompute_w_u` 与 `update_v`,不再物化 h / v_new;intra loop 改为按 H 维并行(原为 Hv),并去除重复的 `key @ key.T` 计算。指令层面:为 decay mask、tril solve、beta/g masking 写 AVX512 专用 kernel(matmul 密集路径用 brgemm + VNNI2 packing);状态(FP32)更新改用 AMX bf16(`C(FP32) += A(BF16) * B(BF16)`);decay mask 与 tril solve 的上三角部分在编译期 mask 掉,减少 FMA;format pack 尽量与 elementwise 运算融合——如 `pack_vnni2` 把 `[K/2, 2, N]` FP32 转为 `[K/2, N, 2]` BF16 的同时就地乘 `exp(g_last)`;l2norm 用 AVX512-bf16 `dpbf16_ps` 做平方和归约。

## 关键改动

```text
sgl-kernel/csrc/cpu/mamba/fla.cpp: chunk_gated_delta_rule kernel 整体重写(模板化 AVX512 kernel 结构)
```

```cpp
// [NOTE] GDN Optimizations on AMX CPU
//   * intra loop: fuse `kkt_solve` and `recompute_w_u` so as to avoid materialize `A`.
//   * inter loop: fuse `recompute_w_u` and `update_v` so as to avoid materialize `h` and `v_new`.
//   * intra loop parallel on H instead of Hv, remove duplicated key @ key.T
//   * update state (FP32) with amx-bf16 where C(FP32) += A(BF16) * B(BF16)
//   * compile time mask out upper triangular part in decay mask and tril solve
```

## 性能声明

"~1.5x speedup against current implementation"(AVX512/AMX CPU 上的 prefill/decode)。PR 未给出具体硬件型号、模型、workload 或数字表格,除该总结性倍数外无其他基准数据。

## 验证与回归风险

准确性测试:`python -m pytest test/registered/cpu/test_mamba.py -v`(PR 未附结果输出)。风险:新增的模板 kernel 均带 `TORCH_CHECK(false, "... scalar path not implemented!")` 的标量回退路径,即非 AVX512(CPU_CAPABILITY_AVX512 未定义)路径会直接报错而非慢速执行;AMX bf16 状态更新的数值容差未在正文说明;PR checklist 的 benchmark/documentation/style 三项未勾选。

## 关联

无(pack 未给出同区域其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
