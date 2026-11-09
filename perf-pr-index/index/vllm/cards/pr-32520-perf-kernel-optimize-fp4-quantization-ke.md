# PR #32520: [Perf][Kernel] Optimize FP4 quantization kernels (SM100F)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-01-25 |
| 分类 | Quantization（tier: major） |
| Label | performance, ready, nvidia |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；pack 中 diff 截断，省略 8 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/32520 |

## 一句话结论

NVFP4 输入量化 kernel 改用 256-bit 全局加载（PTX 8.8，仅 SM100F + CUDA≥12.9）与二维网格划分，microbenchmark 较旧 vLLM kernel 最高快 65%、较 FlashInfer 快 2.33x。

## 动机

FP4（NVFP4）激活量化是 NVFP4 GEMM 前的必经路径，原 vLLM kernel 每线程处理 8 个元素、以 128-bit 加载访问全局内存，全局内存指令数偏多、L2 利用率不足；对大 K 维输入的网格划分也未能充分占用 SM。FlashInfer 的同功能 kernel 在多数 shape 上更快，vLLM 需要自有实现追平并反超。

## 优化点

- 256-bit wide loads：通过 PTX 8.8 的 `ld.global.cg.v8.u32` 一次加载 32 字节（`ld256_or_zero_cg_u32`），`ELTS_PER_THREAD` 从 8 提到 16、`PackedVec` 变为 32 字节对齐；该路径由编译期宏门控（`NVFP4_ENABLE_ELTS16 && CUDART_VERSION >= 12090 && ENABLE_NVFP4_SM100`），不满足条件回退 128-bit 版本。谓词加载对 padding 区域直接置零，替代原 memset/分支。
- 分区方案：网格从一维（行）改为二维（`grid_x` 行 × `grid_y` 列块），`grid_x = min(rows, SM_count*blocks_per_sm / grid_y)`，大 K 输入下提高 occupancy。
- 新增 `is_sf_swizzled_layout` 参数分流两个 kernel：swizzled layout 走原 `cvt_fp16_to_fp4`（含 scale padding 遍历），非 swizzled 走新 `cvt_fp16_to_fp4_sf_major`（row-major `sf_out_rowmajor_u8`，不遍历 padding 行，只迭代 `numRows`）；PACK16 时输出以 64-bit 合并写回（两个 uint32 合成 uint64）。
- 同套改动应用到 fused silu_mul+quant kernel（block 从 1024 调至 512）与 MoE experts 量化路径（`nvfp4_experts_quant.cu`）；数值行为保持（e2m1 转换指令、scale 计算顺序不变，SF 写出提前到输出 scale 计算前）。

## 关键改动

```text
csrc/quantization/fp4/nvfp4_quant_kernels.cu: cvt_fp16_to_fp4 重写 + 新增 cvt_fp16_to_fp4_sf_major（is_sf_swizzled_layout 分流）
csrc/quantization/fp4/nvfp4_utils.cuh: ELTS_PER_THREAD=16 / PackedVec 32B / ld256_or_zero_cg_u32 / fp32_vec16_to_e2m1
csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu: fused silu_mul+quant 同步改造
csrc/quantization/fp4/nvfp4_experts_quant.cu: MoE 量化路径接入新 cvt_warp_fp16_to_fp4
vllm/_custom_ops.py + csrc/torch_bindings.cpp: scaled_fp4_quant 增加 is_sf_swizzled_layout 参数（默认 True）
```

## 性能声明

Microbenchmark（NVIDIA B200，N=8192，K=7168/14336/28672，batch 1–8192，NVFP4 输入量化延迟 us）：新 kernel 在全部测试 shape/batch 上一致快于旧 vLLM swizzle kernel 与 FlashInfer（含 swizzle 变体）；PR 结论为最高较旧 vLLM kernel **快 65%**、较 FlashInfer **快 2.33x**。样例：K=28672, bs=1：vllm-swizzle(prev) 4.279 → vllm(new) 1.661 us（FlashInfer 4.379）；K=7168, bs=4096：15.851 → 9.289 us（FlashInfer 21.608）；K=28672, bs=8192：116.894 → 90.803 us（FlashInfer 167.947）。

E2E（B200，guidellm）：dense Llama-3.3-70B-Instruct-NVFP4（concurrent rate 10，64 in/128 out）输出吞吐 756→785 tok/s（约 +4%）；MoE Qwen3-30B-A3B-NVFP4（flashinfer fp4 moe，rate 128，32 in/128 out）12892.7→13095.9 tok/s（约 +1.5%）。

Accuracy（gsm8k）：dense 0.919→0.921，MoE 0.867→0.870；dense 评测吞吐 output tok/s 3,493.653→5,724.972、总时延 35.874→21.861 s。

## 验证与回归风险

- 单测：`tests/kernels/quantization/test_nvfp4_quant.py`（新增 `test_quantize_to_fp4_padded_no_sf_swizzled` 与 6 个新 pad shape）与 `test_silu_mul_nvfp4_quant.py` 全部通过；`benchmarks/kernels/bench_nvfp4_quant.py` 扩展为 4 provider（vllm/flashinfer × swizzled/non-swizzled）对照并增加 bs=8192。
- 回归风险：优化仅对 SM100F + CUDA ≥ 12.9 生效（其他 arch/toolchain 走原 128-bit 路径）；`scaled_fp4_quant` 签名变更（新增 `is_sf_swizzled_layout`，默认 True 保持旧行为），调用方需显式选择 layout——非 swizzled row-major scale 路径是新面世的行为；谓词加载与 padding 语义改变（不再 memset），由新单测覆盖。

## 关联

- 后续：PR #45739 的描述中引用了本 PR 引入的 `cvt_fp16_to_fp4_sf_major`（row-major sf 输出）作为 non-swizzled 分支的 kernel。
- 同作者同区域：PR #43008（cluster-cooperative topK）；同分类：本批 PR #51473、#45739、#28892、#61457。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
