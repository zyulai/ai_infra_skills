# PR #55062: [Perf] Accumulate Conformer attention scores with baddbmm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Levius-Fubuki |
| 合并 | 2026-09-03 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, multi-modality |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/55062 |

## 一句话结论

Conformer 相对注意力用 baddbmm 累积分数，RTX 4090 CUDA Graph 上几何平均约 1.11x、p50 中位 1.09x。

## 动机

PR Purpose 指出，Conformer relative attention 当前先物化 content scores，再加上 shifted positional scores，然后额外启动一个 in-place scale。这一路径会产生一个 score 大小的中间张量以及独立的 add/scale kernel，存在可消除的开销，因此值得用一次融合的累加运算替代。

## 优化点

用标准 `torch.baddbmm` 把按 `scale` 缩放的 content scores 直接累加进按 `scale` 缩放的 positional scores，内核组合由 `before: {bmm: 2, add: 1, mul_: 1}` 变为 `after: {bmm: 1, baddbmm: 1}`，去掉一个 score 大小中间结果与独立 add/scale kernel。具体实现中保留 `matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2,-1))` 与 `rel_shift`，删去 `matrix_ac` 与 `attn_scores = matrix_ac + matrix_bd; .mul_(scale)`，改为以 reshape 后的 `matrix_bd` 为输入、`beta=self.scale`、`alpha=self.scale` 的 `torch.baddbmm(..., q_with_bias_u.reshape(...), k.transpose(-2,-1).reshape(...))` 再 `view` 回 4 维。PR 强调保留单一可移植 PyTorch 路径，无自定义 kernel、架构门控、设备分支或设备特定调优。

## 关键改动

```text
vllm/model_executor/models/conformer_encoder.py: RelPosMultiHeadAttention.forward 用 torch.baddbmm 替换 matmul+add+mul_。
tests/models/multimodal/test_conformer_encoder.py: 新增回归测试，比对改前实现与改后输出及 post-softmax attention。

  attn_scores = torch.baddbmm(
      matrix_bd.reshape(-1, query_len, key_len),
      q_with_bias_u.reshape(-1, query_len, self.d_k),
      k.transpose(-2, -1).reshape(-1, self.d_k, key_len),
      beta=self.scale,
      alpha=self.scale,
  ).view(batch_size, num_heads, query_len, key_len)
```

## 性能声明

测试条件（RTX 4090 CUDA Graph microbenchmark）：NVIDIA RTX 4090、driver 595.71.05、CUDA 13.0、PyTorch 2.13.0+cu130，无配置改动；`tools/bench_vllm_conformer_baddbmm.py --mode benchmark --batches 1 4 16 --heads 4 8 --sequences 64 128 256 512 1024 --head-dims 32 64 --dtypes float16 bfloat16 --warmups 25 --iterations 200 --repeats 5`。全文 120 行 CUDA Graph 结果：几何平均加速约 1.11x，中位 1.09x，观测最小 1.03x。代表行（p50 Old→New us / p50 speedup / 额外峰值显存 MiB Old/New）：B1 H4 T64 D64 FP16 9.18→8.28 / 1.11x / 0.16→0.12；B1 H8 T128 D32 BF16 1.03x / 1.25→1.00；B4 H8 T512 D32 FP16 170.94→138.72 / 1.23x / 80.03→64.03；B16 H4 T256 D64 BF16 1.07x / 40.03→32.03；B16 H8 T512 D64 FP16 1083.32→955.93 / 1.13x / 320.13→256.13；B16 H8 T1024 D64 BF16 4085.03→3515.28 / 1.16x / 1280.25→1024.25。全部 240 行 eager/CUDA Graph 通过数值校验，零 OOM、零 harness 报告的分配回归。PR 明确说明这是单卡 RTX 4090 的 isolated score-chain 结果，非端到端 Conformer 吞吐；PR 未注明具体模型。

## 验证与回归风险

回归测试 `python -m pytest -q tests/models/multimodal/test_conformer_encoder.py` → 2 passed；该测试实例化真实模块，对序列长度 2 和 8 比较最终输出与 post-softmax attention 与改前实现的一致性，覆盖 masking 与生产 batch-1 positional broadcasting。Lint/pre-commit：`ruff-format`、`ruff-check`、SPDX-header、forbidden-import、新增 `torch.cuda` API 检查、`git diff --check` 本地通过；FP16/BF16 changed-input CUDA Graph replay 与 `torch.compile(fullgraph=True)` smoke check 通过。可能受影响的路径：新路径须与原 `matmul + add + mul_` 数值一致，非连续 Q/K 在两条路径都需相同两次 reshape 拷贝。Limitations：单台 RTX 4090 的 isolated 结果，不对其他架构/backend 作性能声明，完整上游 cross-backend CI 仍需 maintainer `ready`/`verified` label。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
