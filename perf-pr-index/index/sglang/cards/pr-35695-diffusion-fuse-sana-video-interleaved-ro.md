# PR #35695: [diffusion] Fuse SANA-Video interleaved RoPE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-22 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci, diffusion, jit-kernel, mergeable |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及约 9 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/35695 |

## 一句话结论

为 SANA-Video 交错 RoPE 新增配对 BF16 JIT CUDA 内核,H200 上 denoise 每步快约 11%、端到端快约 8.2%。

## 动机

SANA-Video 的交错(interleaved)RoPE 在 eager PyTorch 下由约 14 个 kernel 组成(docs 中 baseline 描述:"about 14 eager kernels"),Q、K 各自独立做 RoPE,每个去噪步都要重复执行。diffusion 推理对延迟敏感,这段逐元素计算是明显的融合目标。难点在于该模型 RoPE 使用 fp64 cos/sin 表,eager TensorIterator 有特定的 fp64→fp32→bf16 转换顺序,朴素融合会破坏 bit-exact 输出。

## 优化点

新增 JIT CUDA 内核 `interleaved_rope_fp64`,一次处理 Q/K 两个张量的配对交错 RoPE:内核内用 fp64 乘加(`__dmul_rn`/`__dsub_rn`/`__dadd_rn`)并刻意保留 eager 的 fp64→fp32→bf16 转换顺序,在 bf16 tie 边界也做到 bit-exact。host 侧用 TensorMatcher 校验形状、4 字节 pair 对齐与输出不别名。Python 侧封装为 custom op `diffusion_interleaved_rope_fp64`,`can_use_interleaved_rope_fp64` 谓词检查 bf16、连续、对齐、fp64 表形状等前置条件。模型侧经 `BitExactFusionGate` 门控接入:首次调用时与 eager 结果比对(`tensors_equal`),一致则永久启用融合路径,不一致或异常则记录并永久回落 eager。

## 关键改动

```text
python/sglang/kernels/jit/csrc/diffusion/interleaved_rope_fp64.cuh: JIT CUDA 内核,fp64 语义的配对 Q/K RoPE
python/sglang/kernels/ops/diffusion/rope/interleaved_rope_fp64_jit.py: custom op 封装与 can_use 谓词
python/sglang/multimodal_gen/runtime/models/dits/sana_video.py: apply_interleaved_rotary_emb_pair 门控接入
test/registered/kernels/ops/diffusion/test_interleaved_rope_fp64.py: bit-exact 单测与谓词用例
test/registered/kernels/benchmark/diffusion/bench_interleaved_rope_fp64.py: microbenchmark(注册 CI)
```

```cpp
const double x1 = static_cast<double>(input[0]);
const double x2 = static_cast<double>(input[1]);
const double x1_cos = __dmul_rn(x1, cos);
const double x2_sin = __dmul_rn(x2, sin);
const double x1_sin = __dmul_rn(x1, sin);
const double x2_cos = __dmul_rn(x2, cos);
// TensorIterator casts the fp64 expression through fp32 before its bf16
// store. Preserving both conversions is required at bf16 tie boundaries.
output[0] = static_cast<T>(static_cast<float>(__dsub_rn(x1_cos, x2_sin)));
output[1] = static_cast<T>(static_cast<float>(__dadd_rn(x1_sin, x2_cos)));
```

## 性能声明

H200,`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`,eager mode,832x480x17,12 denoise steps,guidance 6,seed 42:配对 RoPE microbenchmark 1.4422 ms(eager)→0.1192 ms(JIT,12.1x);denoise latency 0.239–0.245 s/step → 0.212–0.213 s/step(约快 11%);端到端中位数 3.8675 s → 3.5492 s(约快 8.2%);生成视频 SHA256 与 eager 完全一致(`0c1846…a4ae`);ptxas 报告 27 registers、0 spill stores/loads、0 barriers。

## 验证与回归风险

新增 bit-exact 单测(多形状,`torch.equal` 断言)、谓词拒绝用例、CPU 回落单测(test_sana_video.py),PR 报告 19 passed;另含 ruff format/check。首次使用做一次 bit-exact 验证,失败永久 eager 回落,产出不应改变。回归面限于 SANA-Video 的 RoPE 路径;kernel 仅支持 bf16 且要求 fp64 表、连续与对齐,不满足时谓词直接走 eager。pack 中 CI 状态:Base/Extra 通过,AMD ROCm 7.2 run 显示进行中。

## 关联

正文未引用具体 PR 编号;属 sglang-diffusion fused kernels 家族新增条目(docs 表中与 `fused_inplace_qknorm_rope`、`rope_rotate_half`、`ltx2_qknorm_split_rope` 等并列)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
