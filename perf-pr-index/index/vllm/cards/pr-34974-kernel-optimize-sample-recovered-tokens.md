# PR #34974: [Kernel] Optimize sample_recovered_tokens_kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-02-21 |
| 分类 | Speculative Decoding(major) |
| Label | performance, ready, v1 |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 rejection_sampler.py 与新增单测两个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/34974 |

## 一句话结论

rejection sampler 的 recovered-tokens kernel 改为分块 tile 归约并用乘法替代除法,kernel 耗时 159→29 µs(5x+),E2E 输出吞吐 +1%。

## 动机

v1 rejection sampler 的 `sample_recovered_tokens_kernel` 为每个 (request, draft position) program 一次性创建 `tl.arange(0, PADDED_VOCAB_SIZE)` 并整段加载词表(DeepSeek 129k 词表会 pad 到更大的 2 的幂),寄存器压力大、occupancy 低;同时每元素做 `score = prob / q` 除法,而除法比乘法慢。该 kernel 在每次 decode 的 `gpu_model_runner.sample()` 中都会 launch 一次,是大词表 speculative decoding 采样路径的固定开销。

## 优化点

两个层面的改动:一是把词表遍历改成 BLOCK_SIZE=8192 的分块循环,每块只加载一小段,块内做局部 `tl.max(..., return_indices=True)` 归约,再用局部最大值与全局 max 比较更新,寄存器占用从"整张词表"降到"一个块",提高 occupancy;二是 host 侧预先算好 `inv_q = q.reciprocal()`,kernel 内以 `score = prob * inv_q` 的乘法替代除法,归约逻辑保持 argmax 语义不变(`tl.argmax` 只选最大值,无需归一化)。

## 关键改动

```text
vllm/v1/sample/rejection_sampler.py: kernel 改 BLOCK_SIZE 分块 + 局部/全局两级 max 归约;host 预计算 inv_q
tests/v1/sample/test_rejection_sampler.py: 新增 test_sample_recovered_tokens 与原生参考实现逐位对照
```

关键片段(rejection_sampler.py):

```python
max_val = float("-inf")
for v in range(0, vocab_size, BLOCK_SIZE):
    ...
    score = prob * inv_q
    local_max, local_id = tl.max(score, axis=0, return_indices=True)
    if local_max > max_val:
        max_val = local_max
        recovered_id = v + local_id
```

## 性能声明

- Kernel 级:profiling 截图显示 `sample_recovered_tokens_kernel` 约 159 µs → 29 µs(超过 5x)。PR 未注明 profiling 所用硬件与具体 batch/vocab 形状。
- E2E:deepseek-ai/DeepSeek-R1-0528,TP8 + expert parallel,max-num-seqs 16,MTP(num_speculative_tokens=3),ShareGPT 1000 prompts、output 300、并发 16:输出吞吐 1042.81→1053.68 tok/s(+1%),Mean TPOT 14.76→14.62 ms,acceptance rate 58.56%→58.88%。PR 自述 E2E 提升约 1%(该 kernel 每次 decode 仅 launch 一次)。
- 精度:gsm8k(lm_eval,local-completions,并发 16)flexible-extract 0.9583→0.9598,strict-match 0.9553→0.9568。PR 未注明 gsm8k 评测硬件。

## 验证与回归风险

新增 `test_sample_recovered_tokens`:参数化 batch size {1,100} × vocab {100,8192,10000} × max_spec_len {1,3} × 有无 draft_probs,与逐 token 的原生 PyTorch 参考实现(含 per-request generator 的 RNG 状态管理)做 `torch.equal` 精确对照,覆盖 NO_DRAFT_PROBS 分支。风险面:仅影响 v1 rejection sampler 的 recovered-tokens 计算;`prob * inv_q` 与 `prob / q` 在浮点上可能有极小差异,但 argmax 结果经精确单测验证一致;分块循环改变块间扫描顺序不影响 argmax 正确性。

## 关联

同区域(spec-decode 采样/rejection sampling):#14930(v1 rejection sampling 优化)、#11394(FlashInfer 采样 kernel 引入)、#49793(MTP trailing activation 融合)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
