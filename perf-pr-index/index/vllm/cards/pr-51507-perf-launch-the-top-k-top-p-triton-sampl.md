# PR #51507: [Perf] Launch the top-k/top-p Triton sampler kernel with 8 warps

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @BabyDrangoner |
| 合并 | 2026-08-10 |
| 分类 | Attention / Kernel |
| Label | ready, nvidia |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/51507 |

## 一句话结论

top-k/top-p Triton sampler kernel 改用 8 warps 启动,seeded decode 采样 kernel 快 1.84x。

## 动机

`_topk_topp_kernel`(#42191 引入的 Qrita sampler kernel)每个 program 处理一行 logits,并在 `BLOCK_SIZE=8192` 的 tile 中串行扫过整个词表行。Triton 默认 `num_warps=4` 使 8192 宽的 fp32 tile 落到每 lane 16 个元素,tile 延迟(warp 饥饿)直接决定 kernel 延迟。该 kernel 是 CUDA 上 seeded / per-request-generator 采样的热路径(FlashInfer 拒绝 per-request generator):在 Qwen3.6-35B-A3B-FP8 batch-16 decode 上是每步最大的非 GEMM kernel。

## 优化点

改动极小且只在 CUDA 分支生效:GPU 路径启动时传 `num_warps=8`,ROCm 保持默认(64 宽 wavefront,PR 无硬件可测),CPU/XPU 分支不动。参数选择有依据:BLOCK_SIZE 扫描(4k/8k/16k)确认 8192 仍是正确的 tile 尺寸,只有 warp 数不对;选 8 而非 16 是因为 SM120 上 16 warps 在 151k vocab 打平、32k 反而变慢,H20 上 16 虽在 9/12 个点占优(至多 11%)但不稳定,8 是两架构统一的安全选择(如 reviewer 希望也可改成按架构分支)。

## 关键改动

```text
vllm/v1/sample/ops/topk_topp_triton.py: GPU 分支 launch_kwargs 增加 num_warps=8,传入 _topk_topp_kernel 启动
```

```python
else:
    block_size, block_size_trunc = 8192, 4096
    launch_kwargs["num_warps"] = 8
```

## 性能声明

扫描 vocab {32000, 50257, 151936} × batch {8…256} × {k-only, p-only, k+p},fp32 logits,k=20/p=0.95,30 iters 取中位数:RTX PRO 6000 (SM120) 36/36 组合更快,1.2–1.5x;H20 (SM90) 54/54 更快,1.02–1.85x(均值 1.37x);均无回退。e2e:SM120 上 Qwen3.6-35B-A3B-FP8 batch-16 seeded decode,`_topk_topp_kernel` 305.8→165.9 µs/step(**1.84x**);H20 上 `TopKTopPSampler` op 1.16–1.24x,128/128 次 seeded 采样 token ids 完全一致。样例行(k+p,SM120/SM90):vocab 151936 batch 64 为 218.8/236.0→154.2/145.0 µs;batch 256 为 698.4/963.3→526.1/582.3 µs。

## 验证与回归风险

`pytest tests/v1/sample/test_topk_topp_sampler.py`:SM120 上 129 passed/2 skipped,H20 上 135 passed/2 skipped,patch 前后结果一致。正确性方面:top-k 与 top-k+top-p 的 mask 在两架构逐位一致(36/36 与 162/162 例);纯 top-p 存在 pivot 边界翻转(SM120 上 2/36 例单 token;最差 H20 一例 28 token),但对照 float64 nucleus 参考,8 warps 不比 4 warps 差——两者处于同一个先于本 PR 存在的 ~1e-5 kept-mass 近似距离。回归风险很低:仅改 launch 参数,不碰算术;PR 说明中一处此前的失败报告被更正为测试机缺 scipy。

## 关联

kernel 本身由 #42191 引入。无其他同区域 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
