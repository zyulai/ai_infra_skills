# PR #32852: [perf] v1/spec_decode: skip softmax for all-greedy rejection sampling

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @caozuoba |
| 合并 | 2026-01-31 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际仅改动 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/32852 |

## 一句话结论

全 batch 贪心采样时,拒绝采样器跳过全词表 softmax 直接用 logits 取 argmax,输出吞吐 +3.14%、TPOT -7.04%。

## 动机

v1 推测解码的 rejection sampler 之前无条件对 target logits 做 `softmax(dim=-1, dtype=float32)`,这是一个 [num_tokens, vocab_size] 的全词表稠密运算。但当 `sampling_metadata.all_greedy`(整个 batch 都是贪心采样)时,rejection sampling 只需要 `argmax(target_logits)`;由于 `argmax(softmax(logits)) == argmax(logits)`,这个 softmax 在贪心路径上是纯粹的冗余计算,白白增加计算量与显存读写。

## 优化点

把 softmax 从 `forward()` 中无条件执行的位置,延迟到真正需要概率分布的时候:贪心分支先用 `target_logits.argmax(dim=-1)` 启动 `rejection_greedy_sample_kernel`;若 `all_greedy` 则直接返回,softmax 完全不执行。只有当 batch 中存在随机采样请求(`not all_greedy`)时,才在贪心 token 处理完之后对整批 logits 计算 `target_probs` 供后续 rejection sampling 使用。接口层面 `rejection_sample` 的参数从 `target_probs` 改为 `target_logits`,相关 assert 与 shape 检查同步迁移;对随机采样路径而言 softmax 结果不变,行为保持。

## 关键改动

```text
vllm/v1/sample/rejection_sampler.py: softmax 延迟到非贪心路径;all_greedy 时仅对 logits 取 argmax 进贪心 rejection kernel
```

关键代码摘录(节选,<=15 行):

```python
if not sampling_metadata.all_random:
    # Rejection sampling for greedy sampling requests.
    target_argmax = target_logits.argmax(dim=-1)
    rejection_greedy_sample_kernel[(batch_size,)](...)
    if sampling_metadata.all_greedy:
        return output_token_ids

# Compute probability distribution from target logits.
target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
```

## 性能声明

- NVIDIA H800,vs main:Output token throughput +3.14%(8974.16 → 9255.84 tok/s),Mean TPOT -7.04%(39.22 → 36.46 ms),Mean E2EL -3.67%(8557.05 → 8243.21 ms)。
- benchmark 条件:1000 请求,Total input tokens 128000 / Total generated tokens 100000(Peak concurrent requests 1000);pack 未注明被测模型名称与 speculative 配置。

正确性:pytest tests/v1/sample/test_rejection_sampler.py,37 passed。

## 验证与回归风险

- 依赖 tests/v1/sample/test_rejection_sampler.py(37 个用例全部通过)。
- 影响路径:v1 推测解码 rejection sampler 的所有调用。贪心路径结果不变(argmax 等价性);非贪心路径 softmax 计算位置后移但结果一致,并新增 `target_probs.is_contiguous()` 断言。潜在风险集中在混合 batch(贪心+随机)时 logits 引用生命周期与显存占用——softmax 从 forward 顶部移到中间,峰值显存时序略有变化,但张量本身仍会创建。

## 关联

- 同区域推测解码优化:PR #14930(v1 rejection sampler 优化,见索引库同分类卡片);PR 正文未直接列出关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
