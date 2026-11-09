# PR #43390: [Perf] [NVIDIA] fp8_fa_sm120: Use the block-scaled MMA encoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @emre570 |
| 合并 | 2026-10-19 |
| 分类 | Quantization | tier: major |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 3 个文件:`kernel.py`、`plan.py`、`test_fp8_fa_sm120_attn.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/43390 |

## 一句话结论

fp8_fa_sm120 注意力 kernel 改用 block-scaled MMA 编码,GeForce SM120 上 full call 提速 1.47x、输出 bit 级相同。

## 动机

`fp8_fa_sm120` 用 plain FP8 `mma.sync`(E4M3 x E4M3,FP32 accumulate)计算 QK 与 PV 两次 GEMM。在 GeForce SM120 上这条指令以半速率运行:作者在 RTX 5080 上用背靠背 MMA 链、无内存流量实测 raw peak——plain FP8/FP32 accumulate(`QMMA.16832.F32.E4M3.E4M3`)246 TFLOP/s,block-scaled FP8/FP32 accumulate(`QMMA.SF.16832.F32.E4M3.E4M3.E8`)488 TFLOP/s,整两倍。block-scaled 编码(`kind::mxf8f6f4.block_scale`)给每个操作数乘一个 UE8M0 scale(每 32 个 k 元素 1 字节);把所有 scale 字节设为 0x7F(即 2^0)时乘积在数学上完全相同——随机有限 E4M3 输入下,704 个 M 累加器寄存器与 plain 编码 0 个不同。结论:该 kernel 的上限是指令编码而非算法设计。

## 优化点

编译期选择 MMA atom:目标架构支持时(sm_120a/f、sm_121a/f)用 `warp.MmaMXF8Op`,否则回退 `warp.MmaFP8Op`,支持性由 `_block_scaled_mma_supported()` 查 `admissible_archs` 判定。四个 scale fragment(Q、K、P、V)持有常量字节 0x7F,驻留在寄存器——不占 shared memory、无额外 load;两个 `cute.gemm` 调用改为传 `[operand, scale]` 对。`plan.py` 层面公开 `fp8_attention()` 签名不变,内部新增编译期参数 `block_scaled_mma` 并参与 kernel cache key;作者明确它不是用户选项,因为两种编码输出 bit 相同、不存在速度/精度取舍。operand layout、shared memory 布局、softmax、P pack 与 prep 全部不变;寄存器占用 244→250,无 stack spill。

## 关键改动

```text
python/sglang/kernels/ops/attention/fp8_fa_sm120/kernel.py: MmaMXF8Op 选择、unit-scale fragment 构造、两处 cute.gemm 传 [operand, scale]
python/sglang/kernels/ops/attention/fp8_fa_sm120/plan.py: 内部 _fp8_attention(..., block_scaled_mma) 编译期参数并纳入 cache key
python/sglang/multimodal_gen/test/unit/test_fp8_fa_sm120_attn.py: 新增两种编码 bit 相等测试
```

```python
# UE8M0 scale byte: 2^(byte - 127), so 0x7F is exactly 1.0.
UNIT_SCALE_BYTE = 0x7F
...
if cutlass.const_expr(block_scaled_mma):
    mma_op = cute.nvgpu.warp.MmaMXF8Op(
        mQ.element_type, cutlass.Float32, cutlass.Float8E8M0FNU,
    )
else:
    mma_op = cute.nvgpu.warp.MmaFP8Op(
        mQ.element_type, cutlass.Float32, (16, 8, 32),
    )
```

## 性能声明

条件:RTX 5080,H=56,D=128,5 次运行取中位数,同进程同输入。

- S=20440:kernel 56.34 ms→37.66 ms;full call 58.91 ms→40.27 ms(prep 2.60 ms 不变)。
- S=30272:kernel 123.60 ms→83.28 ms;full call 127.26 ms→86.29 ms(prep 3.84 ms 不变)。
- full call 在两个长度均 **1.47x faster**。对照同卡 cuDNN SDPA BF16(S=30272 为 226 ms),该 backend 现在快 **2.6x**(此前 1.8x)。
- 端到端服务吞吐:PR 未提供。

## 验证与回归风险

准确性:输出与 main bit-identical——相同输入(seed 42)对 O 与 LSE 字节做 SHA-256 对比,H=8/S=1052(key mask 路径)、H=56/S=20440、H=56/S=30272 三组全部相等。单测 10/10 通过,含新增 `test_block_scaled_mma_is_bit_equal`(S=1052 与 4096 上强制两种编码逐 bit 相等,输出按 int16、LSE 按 int32 比较)。风险:仅 sm_120a/f、sm_121a/f 编译目标获得新编码,其他目标经 `_block_scaled_mma_supported()` 自动回退 plain 编码,行为不变;causal 输入仍回退 cuDNN,不经过本 kernel;kernel 属 multimodal_gen(diffusion)路径;寄存器 244→250 带来的 occupancy 变化未单独量化。

## 关联

无(正文未列同分类其他 PR;上游依赖为 CUTLASS/cute 的 SM120 block-scaled MMA 支持)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
