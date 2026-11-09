# PR #40785: [JIT] Fuse FP8 KV-cache quantization into the prefix-valid commit kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JiaFengYu |
| 合并 | 2026-10-05 |
| 分类 | Speculative Decoding (tier: major;作用于 DFLASH 验证后的 prefix commit 路径) |
| Label | run-ci, jit-kernel, run-ci-extra, memory-pool |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,6 个文件,含 3 个新测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/40785 |

## 一句话结论

把 FP8 KV cache 的 scale 除法、FP8 转换与 prefix-valid scatter 融合进一个 Triton kernel,该段 GPU 时间 2.99x 提速。

## 动机

DFLASH 验证后,`MHATokenToKVPool.set_kv_buffer_prefix_valid` 负责把被接受的 prefix KV 行写回 KV pool。在 `--kv-cache-dtype fp8_e4m3` 且有 checkpoint KV scales 时,原路径每层要串 5 个 GPU kernel:K 除 scale、V 除 scale、K 转 FP8、V 转 FP8、Triton 字节拷贝/scatter,且中间缩放张量产生完整 HBM round-trip;无显式 scale 时也仍需两次 cast 加 scatter。投机解码每步都要做 prefix commit,这段开销被放大。

## 优化点

新增 Triton kernel `set_kv_buffer_prefix_valid_tiled_fp8`:保留原 (batch, row, tile) 网格结构、commit_lens 提前退出与 loc_2d 间接寻址,改用元素级 offset/stride,接收带类型化的源指针与 FP8 目的指针(uint8-backed KV buffer 的零拷贝 FP8 view)。在 FP32 寄存器里除 scale、先舍回源 dtype 以复现 eager `div_` 语义(tensor scale 用 `div_rn` 正确舍入除法,host scalar 用舍入倒数乘法,两种路径分别对齐 eager 行为),再直接 cast 到 FP8 并 store。全程零中间分配、不写源张量(eager `div_` 会原地改输入)。调度侧收窄:`dtype != cache dtype 且 dtype 为 E4M3 且 K/V scale 均可解析为标量` 才走融合,Python scalar/CPU scalar 直接接受,GPU layer scale 参数用已有 `k_scale_float/v_scale_float` 影子避免 host 同步;不支持的 tensor scale、缺 scale、非 FP8 cache、CPU 执行与非对称 K/V 行均回落原路径。另把 draft model 的 quant_config 传入 DFlashAttention 的 RadixAttention,使 online-FP8 DFLASH draft 能初始化 KV scale 参数与 float shadow。

## 关键改动

```text
python/sglang/kernels/ops/kvcache/cache_move.py: 新增 set_kv_buffer_prefix_valid_tiled_fp8 融合 kernel。
python/sglang/srt/mem_cache/memory_pool.py: 新增 _set_kv_buffer_prefix_valid_impl_fp8、_resolve_fused_scale 与窄化 dispatch。
python/sglang/srt/models/dflash.py: DFlashAttention 的 RadixAttention 传入 quant_config。
python/sglang/test/kernels/prefix_valid.py: 新增,eager oracle 与字节级断言工具。
test/registered/kernels/ops/kvcache/test_prefix_valid_fp8.py: kernel 数值/掩码回归。
test/registered/unit/mem_cache/test_prefix_valid_fp8_pool.py: pool dispatch 与真实 DFLASH 初始化回归。
```

## 性能声明

NVIDIA B200、CUDA 13、PyTorch 2.13.0+cu130、Triton 3.7.1;目标模型 `NousResearch/Meta-Llama-3.1-8B-Instruct`,draft `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`,KV cache fp8_e4m3,draft 量化 fp8,Triton attention backend,batch 4,输出 48 token/请求,接受长度固定 4,CUDA graph 开启。prefix-commit GPU 时间(65 次 commit,profiler):eager fallback 每层中位 14.528 us(K/V 除法 + K/V FP8 cast + byte-copy)vs 融合路径 4.864 us(含前置 source-layout copy),即 2.99x;融合 kernel 本身 1.951 us/层。E2E A/B(10 组交替,每组 100 个 batch-4 请求):request-p50 平均省 0.25 ms,95% bootstrap CI [-0.96, +1.11] ms,跨零,PR 明确不主张端到端显著提升,也不主张 2.99x 端到端。

## 验证与回归风险

新增两个注册测试(kernel 级 + pool 级):与 eager div_->FP8 参考的 FP8 字节级 parity(BF16/FP16/FP32 源)、所有 tile 分支与掩码尾行(row_dim 63/2048/4097)、混合 commit_lens、非连续源、K/V 独立 scale、FP8 边界/NaN/带符号零、未触碰 slot、GPU scale float shadow、缺 scale 与非 FP8 回落、输入不被修改、真实 DFLASH FP8 draft 构造必须走融合路径(用 patch 断言不落入 eager)。本地 CUDA 验证 7 项通过;ROCm 测试已注册(jit-kernel-unit-amd)但本地未跑,E4M3FNUZ 的指针推断与转换在 ROCm 上与 CUDA 不同,且 FNUZ 下 loaded scale 会 x2,风险待 CI 确认;pack 中 CI Base/Extra/AMD 均为 pending(hourglass)。

## 关联

同区域(投机解码 KV 提交/DFLASH):PR #22080(DFlash 投机解码)、PR #22180(spec-ngram followup);KV cache FP8 相关:PR #19103(KV cache FP8 downcast JIT 迁移)。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
