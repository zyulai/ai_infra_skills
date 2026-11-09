# PR #27343: fix(fa3): no NaN embeddings with fa_skip_kv_cache under piecewise CUDA graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27343](https://github.com/sgl-project/sglang/pull/27343) |
| 作者 | jasperjiaguo |
| 合并日期 | 2026-06-15 |
| labels | run-ci, piecewise-cuda-graph |
| 热度 | 👍 0 · 💬 11 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
fa3 的 fa_skip_kv_cache embedding 快路径在 piecewise CUDA graph 下短输入约 40% 返回全 NaN;把 padding 尾部追加为 dummy 自注意段以恢复 varlen 合约。

## 问题现象
`fa_skip_kv_cache` embedding 快路径(#21971 加入)在开启 piecewise CUDA graph 时,约 40% 的**短** embedding 输入返回完全为 NaN 的向量;长输入不受影响。只在 FA3 backend 复现(flashinfer backend 本身已感知 piecewise padding),与 `--prefill-only-disable-kv-cache` 无关(它由 embedding 配置门控)。正文特别说明复现条件:需要 production piecewise 配置(`piecewise_cuda_graph_max_tokens`,即 `ENABLE_PIECEWISE_CUDA_GRAPH`)且 **batch size 1**(每次 forward 一个请求)——请求被 batch 后只有每个 batch 的尾请求贴着 padding 边界,会掩盖单请求失败率。

## 根因
正文披露:`fa_skip_kv_cache` 用 `flash_attn_varlen_func` 在原始 K/V 上做 attention。piecewise CUDA graph 下 model forward 按 **padded token-bucket size** 运行,query 张量 `q` 的行数(B)多于 `cu_seqlens_q` 覆盖的真实 token 数(N)。`flash_attn_varlen_func` 要求 `q.shape[0] == cu_seqlens_q[-1]`;N < B 时该合约被违反,边界的 query block 会破坏**最后一个真实 token** 的输出。embedding 模型用 LAST-token pooling,被破坏的那一行正是返回的 embedding,因此变成 NaN。短输入被 pad 进小 bucket,最后一个真实 token 恰好落在破坏边界上,所以只有短输入受影响。

## 修复方案
把 padded tail `[N, B)` 追加为一个 **dummy、自注意的段**,使 `cu_seqlens = [0, …, N, B]`,`q.shape[0] == cu_seqlens_q[-1]` 恢复成立;真实 token 的段保持不变(attention 不变),padding 行得到有效的隔离 self-attention 而不是读未初始化内存。做法与 flashinfer backend 为 piecewise padding 预留 dummy request 一致。`flashattention_backend.py` 的 `FlashAttentionMetadata` 新增 `fa_skip_cu_seqlens_q`/`fa_skip_max_seqlen_q`(每 forward 构建一次、跨层复用),`_fa_cp_attn` 中构建并传入:

```python
metadata.fa_skip_cu_seqlens_q = torch.cat(
    [cu_seqlens_q, cu_seqlens_q.new_tensor([num_padded_tokens])]
)
```

另:`compiler_interface.py` 的 `AlwaysHitShapeEnv` 增加 `var_to_hint_override` 空映射,兼容新版 torch Inductor。新增回归测试 `test_fa_skip_kv_cache_piecewise_nan.py`(断言无 NaN、与非 piecewise 路径 cosine > 0.99)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:fa_skip_kv_cache 路径的 varlen 元数据构建(仅当 `extend_num_tokens` 可用且小于 q 行数时才加 dummy 段,否则按原样使用);验证:H100 + Qwen3-0.6B embedding 上 NaN 从 81/81 降为 0,与正确路径 cosine mean 0.9999987、min 0.9999946(1000/1000 ≥ 0.99999),即修复后与正确结果逐位等价。

## 关联
- https://github.com/sgl-project/sglang/pull/21971
- https://github.com/sgl-project/sglang/pull/23906

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
