# PR #14930: [V1][Spec Decode] Optimize Rejection Sampler with Triton Kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2025-03-18 |
| 分类 | Attention / Kernel |
| Label | speculative-decoding, ready, v1 |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 5 个文件,3 个被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/14930 |

## 一句话结论

用 4 个 Triton kernel 重写 V1 拒绝采样器,消除 CPU-GPU 同步与冗余数据搬运,ngram SD 吞吐 54.41→64.16 req/s(+18%)。

## 动机

#13933 引入的 V1 拒绝采样器是纯 PyTorch 实现:`draft_token_ids` 以 Python list 逐请求建 tensor 再 `pad_sequence` padding,期间发生 CPU<->GPU 同步;logits/probs 需在 `[num_tokens, vocab]` 与 padded `[batch, max_spec_len, vocab]` 之间来回转换;接受/恢复 token 的计算依赖大量 `cat`/`gather`/scatter 等低效数据搬运。对 ngram 等 draft_probs 为空的路径还要额外构造 one-hot 概率张量,内存与延迟开销都大。

## 优化点

改用 flattend `[num_tokens, vocab_size]` 形状全程处理,不再构造 3D padded 概率张量(显著降显存)。核心是 4 个每请求一 program 的 Triton kernel:`rejection_greedy_sample_kernel`(greedy 请求按 draft vs target argmax 逐位接受,首个不匹配即停)、`rejection_random_sample_kernel`(按 `target_prob/draft_prob >= uniform` 接受,拒绝时取 recovered token,`IS_NGRAM` 时视 draft_prob=1)、`expand_kernel`(按 `cu_num_draft_tokens` 把每请求 temperature 广播展开到 token 维,greedy 用哨兵值替换)、`sample_recovered_tokens_kernel`(用指数分布 q 与 `argmax(prob/q)` 采 recovered token,ngram 路径临时清零 draft token 概率再还原)。softmax 走 `torch.compile` 生成的 `compiled_softmax`(首维 mark_dynamic 防重编译);随机数生成保留 per-request seeded generator 以保可复现。接口改为直接消费 `SpecDecodeMetadata`(cu_num_draft_tokens 等 GPU tensor),实现零 CPU-GPU 同步;删除 `VLLM_USE_FLASHINFER_REJECTION_SAMPLER` env。

## 关键改动

```text
vllm/v1/sample/rejection_sampler.py: 全量重写为 Triton rejection_sample 主路径(4 个 kernel)
vllm/v1/spec_decode/metadata.py: 新增 SpecDecodeMetadata(draft_token_ids/cu_num_draft_tokens 等)
vllm/v1/sample/ops/utils.py: 新增 compiled_softmax(torch.compile softmax)
vllm/envs.py: 移除 VLLM_USE_FLASHINFER_REJECTION_SAMPLER
```

```python
if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
    # Accept.
    token_id = draft_token_id
else:
    # Reject. Use recovered token.
    rejected = True
    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
```

## 性能声明

性能基准:Llama 3.1 8B,ShareGPT,1xH100,temperature 0.1,SD 配置 `--speculative-model "[ngram]" --ngram_prompt_lookup_min 5 --ngram-prompt-lookup-max 5 --num_speculative_tokens 3`:

- main (w/o SD):51.49 req/s;main (w/ SD):54.41 req/s;本 PR (w/ SD):64.16 req/s。
- 即比 main w/o SD +25%,比 main w/ SD +18%。

精度基准:GSM8K,Llama 3.1 8B Instruct,5-shot: Exact match w/o SD 75.7(temp 0)/50.9(temp 1);w/ SD 75.9/51.8,与不开 SD 基本持平。

## 验证与回归风险

PR 正文给出的验证为上述吞吐与 GSM8K 精度对比,未列专门单元测试(diff 中未含新测试文件,另有 3 个文件未进 pack diff)。风险面:输出 padding 语义从 `INVALID_TOKEN_ID` 改名 `PLACEHOLDER_TOKEN_ID`,输出改由 `parse_output` 在 CPU 侧解析;`MAX_SPEC_LEN=32` 成为单步 draft token 上限(超过即 assert 失败);greedy/random 混合 batch 依赖 `is_greedy` 指针为 None 时的重编译行为(作者留有 FIXME);seeded generator 的可复现性依赖 `generate_uniform_probs`/`sample_recovered_tokens` 中按请求跳过 n=0 的逻辑。

## 关联

同区域(spec decode):索引内 PR #13365(n-gram matching 优化,与本文 ngram 用例互补)、PR #25916(EAGLE-3 preamble 修正)、PR #32887(unified parallel drafting)、PR #52228(MRV2 spec decode adaptive verification);前序:PR #13933(原拒绝采样器实现)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
