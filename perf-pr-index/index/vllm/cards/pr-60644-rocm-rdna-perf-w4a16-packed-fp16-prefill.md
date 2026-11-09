# PR #60644: [ROCm][RDNA][Perf] W4A16: packed fp16 prefill dequant and tiles for gfx1151

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stoivonen-amd |
| 合并 | 2026-10-23 |
| 分类 | Model-specific Fast Path |
| Label | performance, rocm, ready, quantization |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/60644 |

## 一句话结论

gfx1151 W4A16 Triton prefill GEMM 改用单指令双 nibble 的 packed fp16 dequant 加新 tile 选型,prefill 延迟最多 -18.8%。

## 动机

`RDNAHybridW4A16LinearKernel` 的 Triton prefill GEMM 在 gfx1151 (Strix Halo) 上,dequant 循环是 VALU-issue-bound:Triton 对每个 4-bit 权重发射一对 `v_and_b16` + `v_or_b16`,每指令只处理一个 nibble;且 fp16 与 bf16 两条路径共用一套按 scalar dequant 调优的 tile,fp16 深层 M 下 `BLOCK_N` 偏小。取代早期 draft #55711。

## 优化点

(1) Packed dequant:利用 fp16 魔数技巧,`(x & 0x000F000F) | 0x64006400` 通过一条 `v_and_or_b32` 把两个 nibble 同时 bitcast 为 `1024 + n` 的 fp16 lane;ExLlama shuffle 布局本就把配对值隔 16 bit,无需额外 shuffle;dequant 减法改为减 `1024 + zp`(fp16 精确表示 2048 以下所有整数,算术仍精确)。仅 fp16 + gfx1151 启用(host 侧判定,以 `PACKED_DEQUANT` constexpr 传入);asymmetric 层只在 M > 256 启用(以下因额外 zero-point load 反而慢 1.2-2.4x)。(2) Tile 选型:新增 `_select_skinny_gfx1151_config`,fp16 且 M > 128(asym 为 256)时用新 tile(深层 M `BLOCK_N` 256 vs 原 64)并设 `num_stages=1`;bf16 保留原 scalar 调优 ladder(`num_stages=1` 在 bf16 asym 上曾损失 15.8% e2e)。

## 关键改动

```text
vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py: _int4_pair_to_fp16x2 内联汇编 + PACKED_DEQUANT 分支 + _select_skinny_gfx1151_config
tests/kernels/quantization/test_rdna_hybrid_w4a16.py: 新增 Triton prefill GEMM 对 float32 oracle 测试(7 组 M x 2 dtype x zp 有无)
benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py: 支持 --dtype float16/bfloat16
```

```python
@triton.jit
def _int4_pair_to_fp16x2(x):
    """Unpack two packed int4 nibbles into a uint32 holding two fp16 lanes,
    each equal to 1024 + nibble, with one ``v_and_or_b32``."""
    mask = tl.full(x.shape, 0x000F000F, tl.int32)
    return tl.inline_asm_elementwise(
        asm="v_and_or_b32 $0, $1, $2, 0x64006400",
        constraints="=v,v,v", args=[x, mask],
        dtype=tl.uint32, is_pure=True, pack=1,
    )
```

## 性能声明

环境:gfx1151 (Radeon 8060S),torch 2.13.0+rocm10.2,Triton 3.8.0;模型 `RedHatAI/Qwen3-4B-quantized.w4a16`(sym)与 `RedHatAI/Qwen3-8B-quantized.w4a16`(asym)。

- E2E prefill(`vllm bench latency`,batch 1,output 1,3 reps 交替):Qwen3-4B fp16 sym 在 64~2048 tokens 全区间快 5~18%(64 tok 34.9→33.0 ms,-5.4%;512 tok 175.7→144.3,-17.9%;2048 tok 782.7→673.5,-14.0%);Qwen3-8B fp16 asym 自 512 tokens 起 -17~-19%(512 tok 325.7→264.5,-18.8%;2048 tok 1346.3→1118.8,-16.9%);bf16 control 持平(±0.4%)。
- 大 batch(batch 128/256,output 1/128):e2e 延迟相对 main 变化 -26.3%~-51.9%;bf16 control ±2.9% 以内。
- Kernel sweep(1248 configs,3 reps 旋转顺序):相对 main 的 geomean,f fp16 sym/asym M>256 为 0.762/0.766;输出 1248/1248 bit-identical。sym fp16 有 46/312 config 慢 >2%(最差 Qwen3-1.7B down_proj M=129-192 慢 13-22%,均承袭自 #55711)。
- GSM8K 5-shot 500 题:±1 题(8B 0.884→0.882,4B 0.848→0.850)。

## 验证与回归风险

`pytest tests/kernels/quantization/test_rdna_hybrid_w4a16.py tests/kernels/quantization/test_w4a16_kernel_selection.py`:gfx1151 上 135 passed(CI 无 gfx11 runner,AMD 量化 shard 跑在 MI300/MI355 上 skip)。新增 prefill GEMM vs float32 oracle 测试覆盖 M=129/256/257 等 packed 切换边界。风险面:改动只影响 gfx1151 fp16(+M 条件)与 tile 表,其他 GPU 与 bf16 不变;新 tile 改变 `BLOCK_K` 从而改变 fp32 累加顺序(实测 bit-identical 但不保证);sym fp16 个别 shape 存在最多 22% 回退(作者声明承袭自 #55711)。

## 关联

同区域(ROCm W4A16/量化):索引内 PR #41394(ROCm native W4A16 kernel)、PR #34709(WvSplitK skinny GEMM)、PR #50578(MLA asm decode);作者列出相关 PR:#57199(HIP decode kernel,不冲突)、#52637(symmetric 权重加载)、#60710(opt-in dense weight cache,同文件但不改 dequant/tile)、前序 draft #55711。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
