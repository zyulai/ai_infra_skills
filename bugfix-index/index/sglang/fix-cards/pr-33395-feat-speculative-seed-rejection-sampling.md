# PR #33395: [Feat] speculative: Seed rejection-sampling draft proposals for deterministic inference

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33395](https://github.com/sgl-project/sglang/pull/33395) |
| 作者 | ajit283 |
| 合并日期 | 2026-10-22 |
| labels | bug, speculative-decoding, run-ci, deterministic, bypass-fail-fast |
| 热度 | 👍 0 · 💬 34 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
EAGLE/NEXTN rejection sampling 的 draft 提议用全局 RNG 抽取,确定性推理下输出仍随 batch 变化;fast_sample 新增按请求种子+位置域分离的确定性路径。

## 问题现象
开启 `--enable-deterministic-inference` 且使用 classic EAGLE/NEXTN rejection sampling 时输出仍不确定:draft 提议由 `fast_sample()` 经全局 PyTorch RNG(`exponential_()`)抽取,而 target 采样与 verify 端 rejection 硬币早已支持请求级种子。复现:Qwen3.5-9B NEXTN rejection sampling、固定采样种子,12 次 changing-batch 试验产生 12 个唯一输出;修复后同样条件 12 次试验只剩 1 个唯一输出(batch 形状变化、prefix batching、bitwise logprob、radix-cache 一致性等确定性测试均通过)。

## 根因
正文披露:draft proposal draw 使用全局 PyTorch RNG,不是 batch-invariant,batch 形状变化与 CUDA graph replay 都会改变输出;#30822 的确定性方案只覆盖 verify 端,draft 侧的种子化留作 follow-up,本 PR 即该 follow-up。

## 修复方案
- `spec_utils.py::fast_sample()` 新增可选确定性路径:以请求采样种子、绝对序列位置、draft step/MTP 层、词表条目为键,用 `murmur_hash32` 域分离(独立 salt `_DRAFT_GUMBEL_SEED_SALT`);哈希经 midpoint binning 转开区间均匀分布后按 float64 指数竞速做分类采样。未启用确定性推理时保留原 RNG 快路径。
- 请求种子与位置贯通 eager 与 CUDA-graph 的 EAGLE/NEXTN worker(含 multi-layer EAGLE 单 CUDA-graph draft-extend),各 runner 新增 `sampling_seed` 输入缓冲。
- 移除「rejection sampling 与确定性推理互斥」的启动校验,改为仅允许 CUDA 设备(依赖 Triton hash kernel)。
- 新增可复现性单测、两个端到端确定性 smoke 测试与 seeded/unseeded 微基准。

注意:pack 的 diff 在 40000 字符处截断(file granularity),其后文件的改动未在素材中展示,此处不推测。

```python
+        hashed = murmur_hash32(
+            salted_seed.to(torch.uint64), positions.to(torch.uint64), col_indices
+        )
+        scores = hashed.to(torch.float64)
+        scores.add_(0.5).div_(2.0**32).log_().neg_().reciprocal_()
+        scores.mul_(probs)
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:`spec_utils` 的 `fast_sample`/`sample_draft_proposal`、eagle_worker_v2 与 multi_layer_eagle_worker_v2 的各 draft 路径、两个 draft CUDA graph runner 的缓冲与参数校验;未启用确定性推理时走原路径,行为不变。确定性路径仅 CUDA 支持;正文 B200 实测 seeded 采样单步变慢 1.48x~3.92x,batch 128/vocab 163840 增量峰值显存 241.25 MiB。

## 关联
- #30822(本 PR 为其中记录的 draft-worker follow-up)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
