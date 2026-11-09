# PR #21971: perf: skip KV cache in FA backend for embedding mode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jasperjiaguo |
| 合并 | 2026-04-13 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files) |
| 链接 | https://github.com/sgl-project/sglang/pull/21971 |

## 一句话结论

embedding 模式下 FlashAttention 后端跳过 KV cache 读写,省掉每层约 19µs 的 store_kvcache 与 prepare_varlen kernel 开销。

## 动机

embedding 模式配合 `--chunked-prefill-size -1` 与 `--disable-radix-cache` 时,每个请求只有单次 prefill、没有 decode 步,KV cache 写入又读出但从不复用,每层浪费约 19µs(`store_kvcache` 约 15µs + `flash::prepare_varlen_num_blocks` 约 4µs)。

## 优化点

`flashattention_backend.py` 新增 `fa_skip_kv_cache` 标志,仅当 `is_embedding && chunked_prefill_size == -1 && disable_radix_cache` 三条件同时满足时启用:(1) 跳过 `set_kv_buffer`,即不做 KV cache 写入;(2) 注意力计算改走 `flash_attn_varlen_func` 直接消费当前层的 raw K/V 张量,替代 `flash_attn_with_kvcache`,完全绕开 paged KV cache 读取;(3) 该模式无 prefix cache,`cu_seqlens_q` 同时用作 Q 与 K 的 varlen 序列边界。`disable_radix_cache` 是必要守卫:若 radix cache 开启,prefix KV 会被跨请求共享,跳过写会让 cache lookup 读到垃圾数据。其他后端不受影响的原因:此前 `save_kv_cache=False` 的 override 放在 `radix_attention.forward()` 层,会破坏 `torch_native`/`triton` 后端(它们跳过写但仍从已空的 cache 读 K/V);本 PR 把 skip 逻辑整体移入 FA 后端自己的 varlen 路径。

## 关键改动

```text
python/sglang/srt/layers/attention/flashattention_backend.py: __init__ 计算 fa_skip_kv_cache 三条件
python/sglang/srt/layers/attention/flashattention_backend.py: forward_extend 跳过 set_kv_buffer;新增 flash_attn_varlen_func(raw K/V)分支
```

核心分支(节选):

```python
result = flash_attn_varlen_func(
    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
    k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
    v=v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_q,
    ...
)
```

## 性能声明

每层消除约 19µs kernel 开销(`store_kvcache` 约 15µs + `prepare_varlen_num_blocks` 约 4µs),条件为 embedding 模式 + 无 chunked prefill + 禁 radix cache。PR 未提供端到端吞吐/延迟 benchmark。

## 验证与回归风险

测试计划:embedding 输出与 baseline 的 cosine similarity > 0.99;cross-encoder 测试确认 `torch_native`/`triton` 后端不再受 `save_kv_cache` override 影响(CI 一项未勾选)。风险:FP8 KV cache descale 在该路径不支持(代码 assert `k_descale`/`v_descale` 为 None);仅覆盖 FlashAttention 后端,其他注意力后端在同类场景下仍走原路径。

## 关联

同区域(卡片库中):#23675(prefill-only 场景禁用 KV cache)同属"跳过无效 KV cache 流量"主题。
