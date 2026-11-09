# PR #36521: [diffusion][kernel] avoid 4D scale-shift autotuning

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-28 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, run-ci, diffusion, jit-kernel |
| Diff | pack 未提供 diffstat(diff 完整可见:scale_shift_triton.py、README、skill 文档、新增 bench 与测试扩展) |
| 链接 | https://github.com/sgl-project/sglang/pull/36521 |

## 一句话结论

4D causal-video scale/shift 路径去掉请求期 Triton autotune,改静态 capped pow2 tile:LingBot-World denoise 冷启动 -21%,稳态 kernel 提速 1.3-2.4x。

## 动机

LingBot-World 每个 transformer block 都调用一次这个 bandwidth-bound kernel;H100 全阶段 trace 中 199 次 Python 调用膨胀为 3,820 次 Triton autotune launch,调优开销主导了第一个 denoise step。且被 autotune 候选(64/128/256/512 tile)在测试过的所有生产形状上输出 bit-identical,capped power-of-two 选择要么最快、要么在测量噪声内——请求期调优纯属浪费。

## 优化点

删除 `_fused_scale_shift_4d_kernel` 的 `@triton.autotune`(5 个 BLOCK_N 配置,key=inner_dim),在 `fuse_scale_shift_kernel` 的 4D 分支改为纯 Python 静态选择:`block_n = max(64, min(512, triton.next_power_of_2(C)))`,`num_warps = 2 if block_n == 64 else 4`,grid 一次算定后作为 constexpr 传入 kernel。同时补齐验证资产:test_modulate.py 新增 4D BF16/FP16 x scale_constant(0/1)的 4 形状正确性用例(与 torch eager 参考对齐,atol=5e-2);新增 registered benchmark `bench_scale_shift_4d.py`(wan/sana_video/longlive/lingbot_world 四个生产形状 + 2 个 CI 缩小形状,`register_cuda_ci` base-b-kernel-benchmark,测冷启动 ms 与 torch/triton 稳态 µs 对比);在 diffusion kernel README 与 benchmark-profile skill 文档中写明"不要在此重新引入请求期 autotuning"的冷启动约束。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/modulate/scale_shift_triton.py: 删 @triton.autotune,静态 capped pow2 tile + constexpr 传入
test/registered/kernels/benchmark/diffusion/bench_scale_shift_4d.py: 新增 registered benchmark
test/registered/kernels/ops/diffusion/test_modulate.py: 新增 4D scale-shift 正确性用例
python/sglang/kernels/ops/diffusion/README.md 与 skill 文档: 记录冷启动约束
```

```python
# Autotuning this bandwidth-bound kernel is much more expensive than
# the launch itself on causal video models. A capped power-of-two
# tile is fastest or within noise across the production hidden sizes.
block_n = max(64, min(512, triton.next_power_of_2(C)))
num_warps = 2 if block_n == 64 else 4
grid = (rows, triton.cdiv(C, block_n))
```

## 性能声明

H100 验证(Native SGLang,LingBotWorldCausalDMDPipeline,832x480x9,4 steps,lossless,seed 42;ABBA 双侧独立 Triton cache,都付真实冷启动成本):

| metric | baseline mean | candidate mean | improvement |
|---|---:|---:|---:|
| denoise | 3909.57 ms | 3083.09 ms | 21.14% |
| request E2E | 12416.27 ms | 11467.29 ms | 7.64% |

4 个 ABBA MP4 SHA256 一致(输出无差异)。新隔离 Triton cache 下生产形状 [1, 4680, 5120] 首次调用 1338.47 ms → 714.77 ms。registered benchmark 稳态 kernel 提速:Wan 2.36x、SANA Video 2.29x、LongLive 1.33x、LingBot-World 2.24x。LingBot-World V2 兄弟 checkpoint(robbyant/lingbot-world-v2-14b-causal-fast-diffusers)fresh-cache A-B-B-A:denoise 3882.74 → 3015.31 ms(22.34% faster),request E2E 12953.80 → 12168.34 ms(6.06% faster),MP4 字节一致。LingBot causal DMD 不支持 breakable CUDA graph,PR 不做 BCG 性能声明。

## 验证与回归风险

`test_modulate.py` 993 passed;`test_lingbot_causal_denoising.py` 28 passed;rebase 后 4D selection 16 passed;pre-commit 全过。输出等价性用 MP4 SHA256 双重锁定(主 checkpoint 与 V2 checkpoint 各一组)。回归面:`fuse_scale_shift_kernel` 的 4D `[B, F, 1, C]` 路径(要求 `L % F == 0`);2D/其他维度路径不变;静态 tile 选择覆盖 C<=512 的 pow2 内插值,C 超过 512 时封顶——PR 声称在测试过的生产隐藏维度上 bit-identical,更大 C 未在验证列表中;BCG 组合静态不支持。

## 关联

同区域:#29664(同作者对 jit_kernel/diffusion 的清理)、#26045(group_norm_silu 融合,同属 diffusion kernel 融合方向);pack 正文未直接列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
