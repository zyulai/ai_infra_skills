# PR #20778: [FlashAttn] Add fused triton kernel for normal_decode_set_metadata

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @libowen2121 |
| 合并 | 2026-03-22 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,2 个文件,含 1 个新测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/20778 |

## 一句话结论

用融合 Triton kernel 替代 FlashAttention decode 元数据准备的 4-5 个串行 kernel,隔离基准提速 4.78x,decode 元数据段 CPU 开销显著下降。

## 动机

`flashattention_backend.py` 的 `normal_decode_set_metadata` 每次 decode 都要准备 attention 元数据,原本由 4-5 个串行 CUDA kernel 组成:(1) `cache_seqlens = seq_lens + seq_len_delta` 的 int64->int32 转换、(2) `cu_seqlens_k` 前缀和、(3) `req_to_token` 二维 gather、(4) `page_table = page_indices // page_size` 整除、(5) SWA 时的 full->swa 页表转换。代码中本有 `TODO: fuse these kernels`,且注释说明 `torch.compile` 在投机解码下反而更慢,一直未融合。每步 decode 都执行这段,launch 开销与中间张量读写值得消除。

## 优化点

两个 Triton kernel 按场景分流:(1) `_fused_metadata_kernel_ps1_no_swa` 专门优化最常见路径(page_size=1、无 SWA),直接把 token 索引当页索引写 page_table;(2) `_fused_metadata_kernel_general` 支持任意 2 的幂 page_size 与可选 SWA,用移位替代乘/除(`col_idx << SHIFT`、`page_index >> SHIFT`),SWA 时在 kernel 内做 `full_to_swa_index_mapping` gather。两个 kernel 都在 (0,0) block 内串行做前缀和并写出 `cache_seqlens_int32` 与 `cu_seqlens_k`,其余 block 按 (batch, 256/128 列块) 并行做 gather 与 store,load/store 使用 `.cg` cache modifier。入口断言 page_size 必须为 2 的幂,并对输入做 contiguous 化。整体由 4-5 个 kernel 降为 1-2 个。

## 关键改动

```text
python/sglang/srt/layers/attention/flashattention_backend.py: 新增两个 Triton kernel 并重写 normal_decode_set_metadata,删除原 5 步 eager 实现。
test/registered/attention/test_normal_decode_set_metadata.py: 新增,16 个用例(注册 CUDA CI stage-b)。
```

前缀和与 gather 融合的核心结构:

```python
if pid_b == 0 and pid_c == 0:      # 单 block 串行前缀和
    acc = 0
    for idx in range(B):
        val = (seq + seq_len_delta).to(tl.int32)
        tl.store(cu_seqlens_k + idx * stride, acc); acc += val
# 其余 block:page_index = req_to_token[row, col] (gather)
#            page_table[i, col] = page_index >> SHIFT
```

## 性能声明

隔离 kernel 基准(1000 次迭代,NV-H200,bs 32、page_size 1、max_ctx 8192、max_pool 1024、seq_delta 0):相对原实现 4.78x speedup。E2E(`sglang.bench_serving`,meta-llama/Meta-Llama-3.1-8B-Instruct,NV-H200,input=output=seqlen):整体改善约 0.02%-1.2%,如 bs16/seqlen 2048 为 7014.71 -> 6978.49 ms,bs32/128 为 513.51 -> 504.52 ms,bs64/512 为 2337.13 -> 2321.75 ms,bs32/64 出现 301.44 -> 302.48 ms 的轻微回退;差异属 E2E 噪声量级,主要收益在 kernel 段。

## 验证与回归风险

新增注册单测与参考实现逐位对比(`torch.equal`):page_size 1/16/64、有无 SWA、seq_len_delta、batch 1-64、2 的幂 page_size 1-128 全覆盖、变化剧烈的序列长度混合;另跑通 `test_flashattn_backend.py` 全部 7 项(decode/extend/page>1/prefix/投机 draft metadata)。回归风险:(1) 要求 page_size 为 2 的幂(assert),非 2 幂 page 配置会直接报错而非回落;(2) 前缀和在单 block 内对 B 串行循环,超大 batch 时该段是 O(B) 串行;(3) 输入被强制 `.contiguous()`,非连续大张量会引入额外拷贝;(4) SWA 路径依赖 `SWAKVPool.full_to_swa_index_mapping` 的 contiguous 布局。

## 关联

同区域(FlashAttention backend 元数据优化):PR #17554(kernel 优化 decoding metadata in NSA)、PR #14781(NSA backend metadata 优化)。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
