# PR #30897: [NVFP4][Perf] Tune NVFP4 input quant kernel for small batch size

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-12-21 |
| 分类 | Quantization |
| Label | performance, ready, nvidia |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/30897 |

## 一句话结论

优化 NVFP4 输入量化 kernel 的小 batch 场景:grid 按 128 行 tile 对齐扩展 + SF offset 换算改位运算,bs=1 decode 吞吐 dense 模型 +25%。

## 动机

作者发现 NVFP4 输入量化算子在 dense 模型小 batch(尤其 bs<32)时严重拖慢性能。参考 FlashInfer 实现里 "increase the grid size for small M" 的做法:swizzled SF layout 要求按 128 行 tile 覆盖 padding 行,而 vLLM 的 grid 之前按真实 m 截断,小 M 时 SM 占不满。另外 `cvt_quant_to_fp4_get_sf_out_offset` 内循环存在冗余计算,nvcc 未把除法优化成位运算。

## 优化点

三点改动:一是新增 `computeEffectiveRows(m)`(把 m 向上 round 到 128 行 tile),grid 上限从 `min(m, SM*blocksPerSM)` 改为 `min(effectiveRows, ...)`,小 M 时启动更多 block 覆盖 swizzled SF 的 padding 区域,这是 bs<32 提速的主因。二是 SF 输出 offset 计算重写:调用方(kernel 入口)预计算一次 `numKTiles = (numCols + 63) / 64` 作为常量传入,函数内把 mTileIdx/outerMIdx/innerMIdx/kTileIdx/innerKIdx 的除法取模全部换成移位与掩码,最终 offset 用一个 `(mTileIdx*numKTiles + kTileIdx) << 9 | (outerMIdx<<4) | (innerMIdx<<2) | innerKIdx` 位或表达式合成,消除内循环每线程的多次除法;experts 量化 kernel 中原先每 expert 重复计算 `numCols_padded/numCols_SFout` 的逻辑也被 numKTiles 取代。三是 `quant_impl` 里每次调用都 `cudaDeviceGetAttribute` 查 SM 数的 TODO 项改为走缓存的 `get_device_attribute`,并删除不再使用的 `invokeFP4Quantization` 包装。

## 关键改动

```text
csrc/quantization/fp4/nvfp4_quant_kernels.cu: computeEffectiveRows 调整 grid,直连 kernel 启动
csrc/quantization/fp4/nvfp4_utils.cuh: cvt_quant_to_fp4_get_sf_out_offset 位运算重写(签名改收 numKTiles)
csrc/quantization/fp4/nvfp4_experts_quant.cu: MoE experts 量化 kernel 同步 numKTiles 与 SFout 偏移
csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu: silu-mul 融合量化 kernel 同步改动
benchmarks/kernels/bench_nvfp4_quant.py: 新增 vllm vs flashinfer 量化 microbenchmark 与精度对比
```

## 性能声明

端到端 bs=1 decode(`vllm bench serve --input-len 10 --output-len 1000 --num-prompts 1`):dense Qwen3-8B-NVFP4 输出吞吐 220 → 257 → 276 toks/s(+25%);MoE Qwen3-30B-A3B-NVFP4 输出吞吐 137 → 145 toks/s(+6%)。gsm8k 精度:dense 0.854(前后一致),MoE 0.880 → 0.894(吞吐 6662.9 → 6617.8 output tokens/s)。硬件未在文中明说,benchmark 脚本要求 Blackwell(compute capability 10.0)。

## 验证与回归风险

- 新增 `benchmarks/kernels/bench_nvfp4_quant.py` 带 `--accuracy` 模式断言 vLLM 与 FlashInfer 量化输出一致(M=1/1024, K=4096);gsm8k 端到端精度见上。
- 风险:SF offset 位运算依赖所有除数为 2 的幂(CVT_FP4_SF_VEC_SIZE=16、tile 尺寸 32/4 等),layout 假设写死;grid 按 128 行向上取整会为 padding 行启动空转 block,大 M 时因 `min(effectiveRows, SM*blocksPerSM)` 上限不变,不影响正常路径;MoE experts kernel 的 SFout 基址改为 `* numKTiles`,需与 CUTLASS GEMM 期望的 swizzled scale 布局保持一致。

## 关联

- 同区域:#18844(scaled FP8 quant 提速,同类小 kernel 优化)、#23696(B200 MXFP4 CUTLASS MoE)、#34664(MXFP8 Marlin)。
- 上游参照:flashinfer-ai/flashinfer quantization.cu 的 effective-row grid 逻辑。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
