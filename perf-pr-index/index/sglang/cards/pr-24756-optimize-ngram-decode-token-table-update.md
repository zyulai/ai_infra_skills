# PR #24756: Optimize ngram decode token table update

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-06 |
| 分类 | Speculative Decoding (tier: major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files) |
| 链接 | https://github.com/sgl-project/sglang/pull/24756 |

## 一句话结论

为 ngram 投机解码的 token 表更新增加 decode 专用 kernel,大 batch 下每步更新从 ~286µs 降到 ~5µs(+98%)。

## 动机

ngram 路径中 `ModelRunner.maybe_update_ngram_token_table` 在每个 decode 步都要把每请求 1 个新 token 写入 ngram embedding token 表;原 `update_token_table` general kernel 按请求分块、按 `req_lens` 循环写并处理 ignore tokens,decode 场景(req_lens 恒为 1)下这些通用逻辑使开销随 batch 规模线性增长,大 batch 时成为每步固定税。PR 正文无独立动机章节,以上来自 Summary 与 benchmark 数据的指向。

## 优化点

新增 `UpdateTokenTableDecodeKernel` CUDA kernel:grid-stride 循环,每个线程处理一个请求,一步完成 `ne_token_table[row_indices[req] * max_context_len + column_starts[req]] = tokens[req]`,不再需要 `req_lens` 与 `ignore_tokens` 输入,block 数上限 1024×256 线程。`ModelRunner.maybe_update_ngram_token_table` 在 decode 快路径改调 `update_token_table_decode`,同时省去每次调用构造 `torch.ones_like` 的 req_lens 张量;general `update_token_table` 保留给 prefill / 变长请求。附带修正:两个 kernel 的行/列索引计算统一升级为 int64(`row_indices` 改 `const int64_t* __restrict__`),避免 `row_indices * max_context_len` 在大表下 int32 溢出。

## 关键改动

```text
python/sglang/jit_kernel/csrc/ngram_embedding.cuh: 新增 UpdateTokenTableDecodeKernel;索引 int64 化
python/sglang/jit_kernel/ngram_embedding.py: 暴露 update_token_table_decode wrapper
python/sglang/srt/model_executor/model_runner.py: decode 路径改调 update_token_table_decode
python/sglang/jit_kernel/benchmark/bench_ngram_update_token_table.py: 新增 general vs decode 基准(注册 CI)
```

## 性能声明

H200 上 CUDA event median,general 路径 → decode 快路径(NVIDIA H200,基于 origin/main `9474f8e98` 的本地 merge-preview,kernel 单独计时、无 CUDA graph;token 表 [bs+8, 4096] int32):

| batch size | general | decode fast path | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 5.184 us | 4.640 us | +10.49% |
| 32 | 5.728 us | 4.608 us | +19.55% |
| 128 | 7.872 us | 4.672 us | +40.65% |
| 512 | 15.552 us | 4.960 us | +68.11% |
| 1024 | 30.176 us | 4.960 us | +83.56% |
| 2048 | 83.744 us | 5.024 us | +94.00% |
| 4096 | 286.368 us | 5.056 us | +98.23% |

## 验证与回归风险

`test_ngram_embedding.py` 新增 `test_update_token_table_decode_matches_general`(bs 1/2/17/128/1024,随机 row_indices,与 general kernel 输出 `atol=0, rtol=0` 逐位一致);PR 报告 5 个测试通过。风险:decode 快路径不走 ignore_tokens 过滤,仅当调用点保证 decode 步无 ignore token 时与 general 等价;int64 索引改动同时作用于 general kernel;PR 页面 CI 状态显示 Base/Extra 均失败(x)。

## 关联

姊妹 PR:#24757(同作者同日,Optimize ngram decode id computation);同区域:#22180(spec ngram follow-up fixes)、#22077(dflash 投机解码)。
