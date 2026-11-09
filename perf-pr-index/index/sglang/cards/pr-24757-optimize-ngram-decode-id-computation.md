# PR #24757: Optimize ngram decode id computation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-02 |
| 分类 | Speculative Decoding (tier: major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files) |
| 链接 | https://github.com/sgl-project/sglang/pull/24757 |

## 一句话结论

ngram embedding 的 n-gram id 计算增加 decode 专用 thread-per-output kernel,decode 路径省掉 cumsum,大 batch 提速最高 +86.57%。

## 动机

ngram 投机解码的 over-embedding 路径在 decode 模式下仍走 general `compute_n_gram_ids`:先 `torch.cumsum` 算 req_lens 前缀和,再由每请求一个 block 的 kernel 串行处理 token 循环。decode 时每请求只贡献 1 个 token,这些通用逻辑(前缀和 launch、逐 token 偏移计算)成为每步固定开销。

## 优化点

新增 `ComputeNGramIdsDecodeKernel`:grid-stride 循环,每线程负责一个 (request, config) 输出(总输出 = batch_size × (ne_n-1)×ne_k),直接从 token 表读当前 token 及往前至多 `n_idx+2` 个 token 计算多项式哈希 n-gram id,负 token 值(未填充槽位)即 break;不需要 tokens 数组与 `exclusive_req_len_sums`。`NgramEmbedding.forward` 的 decode 分支改调 `compute_n_gram_ids_decode` 并完全跳过 `torch.cumsum(req_lens)`;prefill/extend 保持 general 路径不变。kernel 入口用 TensorMatcher 做形状/设备/dtype 校验,int64 行偏移防溢出。

## 关键改动

```text
python/sglang/jit_kernel/csrc/ngram_embedding.cuh: 新增 ComputeNGramIdsDecodeKernel(每输出一线程,免前缀和)
python/sglang/jit_kernel/ngram_embedding.py: 暴露 compute_n_gram_ids_decode wrapper
python/sglang/srt/layers/n_gram_embedding.py: decode 分支走新 kernel 并跳过 torch.cumsum
python/sglang/jit_kernel/tests/test_ngram_embedding.py: 新文件,decode vs general 一致性测试
python/sglang/jit_kernel/benchmark/bench_ngram_compute_decode.py: 新增基准(注册 CI)
```

## 性能声明

H200 forward-like benchmark,base `torch.cumsum + compute_n_gram_ids` → decode fast path,所有行 `max_abs_diff = 0`(NVIDIA H200,基于 origin/main `90bd54005` 的本地 merge-preview;基准参数 NE_N=8、NE_K=2、vocab 32000、max_context 1024,无 CUDA graph):

| batch size | base | this change | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 22.496 us | 10.176 us | +54.77% |
| 128 | 23.200 us | 10.208 us | +56.00% |
| 1024 | 35.232 us | 10.048 us | +71.48% |
| 2048 | 48.496 us | 10.096 us | +79.18% |
| 4096 | 74.800 us | 10.048 us | +86.57% |

另一组仅计 `compute_n_gram_ids`(不含 cumsum)的基准:batch 4096 时 62.592 us → 8.800 us。

## 验证与回归风险

新增 `test_compute_n_gram_ids_decode_matches_general`(bs 1/2/17/128/1024,`atol=0, rtol=0` 逐位一致),PR 报告 5 tests passed;CI Extra 通过,Base 未执行(no_entry_sign)。风险:decode 路径假设每请求恰 1 个 token,哈希参数(ne_weights/ne_mods)必须与 general 完全一致,正确性依赖该测试兜底。

## 关联

姊妹 PR:#24756(同作者,ngram decode token table update,在其后合并);同区域:#22180(spec ngram follow-up fixes)。
