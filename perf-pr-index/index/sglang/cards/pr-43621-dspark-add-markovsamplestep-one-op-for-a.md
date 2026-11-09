# PR #43621: [DSpark] Add `MarkovSampleStep`, one op for a markov draft step's add and token pick

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-10-11 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | deepseek, jit-kernel |
| Diff | pack diffstat 缺失;diff 为新 markov_sample.py、删除 sharded_greedy.py、DSpark 模型接线与新测试文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/43621 |

## 一句话结论

把 DSpark Markov draft 步的 bias 加法、舍入与 token 选择合并为单个 op(kernel 内 Philox 噪声),采样行快 2.0–2.3x,每步只需一次 collective。

## 动机

DSpark 的 Markov draft 每步要完成:把 step bias 加到 base logits、按 head 的 step dtype 舍入、再为每行选 token(温度 0 取 argmax,否则 Gumbel-max 采样)。此前 greedy 路径走 `sharded_greedy.py` 的专用 kernel,采样路径则是另一条更慢的实现;噪声还需外部张量。`MarkovSampleStep` 把这一整步收敛为一个 op(叠加在 #43620 之上)。

## 优化点

Triton 与 torch 双实现,单 op 覆盖 add + round + pick。噪声为 kernel 内计算的 Philox4x32-10:key 取 server seed,counter 由 (global token id, row, step, per-block offset) 构成——不需要噪声张量、不触碰 torch generator,且每个 TP rank 从相同输入抽出相同值(全局 token id 保证 sharded 与 full vocab 抽样一致)。TP-sharded vocab 下两种模式各只需一次 collective:无 kept logits 时每个 rank 把自己的 shard 按 1024-wide block 归约成每 block 一个 (value, index) 对、只 all-gather 这些对再跨 rank 取优;有 kept logits 时 step logits 一次 gather、每个 rank 从全 vocab 选择。无采样行时编译 greedy-only 变体(不读 temperature、无噪声代码)。替换 `sharded_greedy.py`,DeepSeek-V4.1 greedy graph tail 改调该 op。行为不变:greedy picks 仍是相同 argmax、相同 lowest-index tie 规则。

## 关键改动

```text
python/sglang/kernels/ops/speculative/dspark/markov_sample.py: 新增 MarkovSampleStep(partial/finish 两段 Triton kernel + torch 参考)
python/sglang/kernels/ops/speculative/dspark/sharded_greedy.py: 删除,由 MarkovSampleStep 的 greedy-only 变体取代
python/sglang/srt/models/deepseek_v4_dspark.py: sample_block_greedy_fused 改调 MarkovSampleStep.triton(has_sampled_rows=False)
test/registered/kernels/ops/speculative/test_dspark_markov_sample.py: 新增 9 项测试
```

```python
# 采样行的 Gumbel-max key(kernel 内生成噪声,无需噪声张量)
w0, w1, w2, w3 = tl.philox(KEY, quad.to(tl.uint32), (lane + row).to(tl.uint32),
                           (lane + STEP).to(tl.uint32),
                           ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
                           PHILOX_ROUNDS)
...
uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
exp_noise = -libdevice.log1p(-uniform)
key = value / temperature - tl.log(exp_noise)
```

## 性能声明

per launch、对比被替换的 kernel(GB300,torch 2.14.1、Triton 3.8.0):greedy-only 变体不慢于旧 sharded greedy step;采样行比旧 sampling path 快 **2.0–2.3x**;1024-wide block 把 129280-token vocab 上的一行采样步从 5.6 us 降到 2.4 us。无端到端吞吐数据。

## 验证与回归风险

`test_dspark_markov_sample.py` 9 passed on GB300:Tr​iton 对 torch reference(bf16/fp32 组合,greedy 与采样行混合)、greedy-only 变体对 mixed kernel、NaN/inf 特殊值、跨 block tie 取低 index、strided rows、tempered-softmax 分布(0.5/1.0/2.0 温度,频率差 <0.01)、noise keying(offset/step 变化改变抽样)、sharded 与 full vocab 在两种 gather 模式下 token/logits 一致。风险:自定义 Philox keying 与 torch 全局 generator 语义不同(确定性可复现但非 torch RNG);finish kernel 依赖 NVLink push transport 的 −0 sentinel 约定(`idx &= _NO_INDEX` 恢复 ID 0);采样行在近 tie 处 Triton 与 torch 可能因 last-ulp 差异分叉(测试要求 ≥0.95 agreement)。

## 关联

#43620(本 PR stacked 于此)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
