# PR #35698: [diffusion] Fuse LTX-2.5 decoder 3D RoPE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-21 |
| 分类 | Attention / Kernel |
| Label | documentation, run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/35698 |

## 一句话结论

LTX-2.5 decoder 3D RoPE 用缓存轴表 + 单 launch 成对 Q/K JIT kernel,H200 全 decoder 组件中位延迟快 17.0%。

## 动机

LTX-2.5 decoder 的 3D RoPE 在 eager 路径下有两个开销:每个 attention block 都重复构建 temporal/height/width 三组 cos/sin 角度表;Q、K 分别走 fp32 乘加、stack、cat 等一串小 kernel。该 RoPE 是多块 decoder 的公共路径,合并成对旋转 kernel 并跨块缓存表是直接收益点。

## 优化点

两层优化:(1) 表缓存——`_ROPE_TABLE_CACHE` 按 (num_frames, height, width, rope_dim_split, base, device) 为 key 缓存紧凑 fp32 轴表(上限 16 条),同一 decoder grid 的所有 attention block 复用同一份表,复用性有单测断言 `first is second`;(2) 成对 BF16 JIT CUDA kernel——一个 launch 同时旋转 Q 与 K,`rotate_pair` 用 `__fmul_rn/__fsub_rn/__fadd_rn` 分步舍入,保持与 eager fp32 表达式完全相同的舍入顺序,保证 bit-exact;kernel 按 pair 维度 grid-stride,fp32 cos/sin 表经 `SGLANG_LDG` 读取。安全机制:`can_use_ltx25_decoder_rope` 谓词检查 6 维 shape、bf16、contiguous、4 字节对齐、无别名、表 dtype/shape;`BitExactFusionGate` 首次运行时对 eager 输出做逐位校验,通过后永久启用,失败则永久回退 eager。

## 关键改动

```text
python/sglang/kernels/jit/csrc/diffusion/ltx25_decoder_rope.cuh: 新增成对 Q/K 3D RoPE CUDA kernel(按 t/h/w 轴查表)
python/sglang/kernels/ops/diffusion/rope/ltx25_decoder_rope_jit.py: JIT 装载器、custom op 注册与 can_use 谓词
python/sglang/multimodal_gen/runtime/models/decoders/ltx_2_5_diffusion_decoder.py: 轴表跨块缓存 + forward_pair 快慢路径
test/registered/kernels/ops/diffusion/test_ltx25_decoder_rope.py: bit-exact 与谓词拒绝测试
test/registered/kernels/benchmark/diffusion/bench_ltx25_decoder_rope.py: 配套 benchmark
```

关键片段(舍入顺序对齐 eager):

```cpp
const float even_cos = __fmul_rn(even, cos);
const float odd_sin  = __fmul_rn(odd,  sin);
const float even_sin = __fmul_rn(even, sin);
const float odd_cos  = __fmul_rn(odd,  cos);
output[0] = static_cast<T>(__fsub_rn(even_cos, odd_sin));
output[1] = static_cast<T>(__fadd_rn(even_sin, odd_cos));
```

## 性能声明

PR 正文,H200:
- 配对 RoPE 单 kernel benchmark:eager → JIT,stage 0: 0.9381 ms → 0.0950 ms(9.87x);stage 4 tile: 2.5454 ms → 0.2467 ms(10.32x);stage 5 tile: 9.3790 ms → 0.8643 ms(10.85x)。
- 全 decoder 组件 benchmark(默认 960x544x121 geometry、全部 24 blocks、tiled decode、NATTEN `na3d`):中位延迟 5379.54 ms → 4463.80 ms(17.0% faster);输出 SHA256 逐位一致;峰值分配显存基本不变(7,422,371,328 → 7,422,650,880 bytes);ptxas sm_90a:30 registers、0 spill、0 barriers。
- 限制:HF checkpoint gated、基准机无凭证,组件 benchmark 用确定性初始化的全尺寸 decoder 权重,未跑官方发布权重。

## 验证与回归风险

新增 pytest 测试(test_ltx25_decoder_rope.py bit-exact + 谓词拒绝、test_import_surface、test_ltx2_5_config.py 的 CPU fallback 与表共享用例),PR 报告 55 passed, 5 subtests passed;另做 ruff check/format 与 git diff --check。风险面:快路径仅限 bf16、6 维 contiguous、fp32 表且对齐的输入,谓词不满足或首验不逐位一致自动回退 eager,对 LTX-2.5 之外模型无影响;注意 PR CI 状态显示 PR Test (Base) 与 (Extra) 均为失败(x),AMD ROCm 7.2 为 pending,合并时 CI 未全绿。

## 关联

同属 diffusion fused kernel 体系(same registry:`ltx2_qknorm_split_rope`、`hunyuan_qkv_rope_pack`、`fused_inplace_qknorm_rope` 等已有条目,本卡 PR 在同一 `fused_kernels.mdx` 表中注册 `ltx25_decoder_rope`);同批 diffusion 相关:PR #28527([Diffusion][CPU] AMX 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
