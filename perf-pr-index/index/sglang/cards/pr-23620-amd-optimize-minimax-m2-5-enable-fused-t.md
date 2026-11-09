# PR #23620: [AMD] Optimize MiniMax-M2.5 - enable fused Triton kernel for FP8 KV cache write in aiter decode path

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yctseng0211 |
| 合并 | 2026-04-25 |
| 分类 | Quantization(major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示仅改动 aiter_backend.py 一处分支(+17 行) |
| 链接 | https://github.com/sgl-project/sglang/pull/23620 |

## 一句话结论

AMD aiter decode 路径上把 FP8 KV cache 写入的 dtype cast 与 paged store 两次 kernel launch 融合为一次 Triton kernel,MiniMax-M2.5 在 MI355X 上低并发吞吐最高 +5.9%。

## 动机

在 AMD GPU 上启用 FP8 KV cache(`--kv-cache-dtype fp8_e4m3`)与 unified attention 时,decode 阶段的 KV cache 写入此前需要两次独立的 kernel launch:先做 bf16→fp8 的 dtype cast(`float8_copy_kernel`),再做 paged store(`store_kvcache`)。两次 launch 带来额外开销,在 decode 这种短 kernel 密集的路径上值得消除。

## 优化点

在 `AiterAttnBackend.forward_decode` 中新增一个分支:当 `use_triton_unified_attention` 开启且 kv_cache_dtype 为 fp8 时,改用已有的 `launch_reshape_and_cache_flash` Triton kernel(此前已服务于 SWA 模型),把 bf16→fp8 cast 与 paged 写入融合到单次 kernel launch 中。实现上从 `token_to_kv_pool.get_kv_buffer(layer.layer_id)` 取出 k/v cache buffer,把当前 step 的 k/v view 与 cache buffer 的 view 一起传入该 kernel,配合 `forward_batch.out_cache_loc` 完成写入;非 FP8 或非该后端条件的路径保持原 `set_kv_buffer` 逻辑不变。

## 关键改动

```text
python/sglang/srt/layers/attention/aiter_backend.py: forward_decode 新增 FP8 非 SWA 分支,
  用 launch_reshape_and_cache_flash 融合 cast + paged write,消除独立的
  float8_copy + store_kvcache 开销
```

关键摘录(diff):

```python
elif self.use_triton_unified_attention and self.kv_cache_dtype == fp8_dtype:
    token_to_kv_pool = forward_batch.token_to_kv_pool
    k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
    launch_reshape_and_cache_flash(
        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
        k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim),
        v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim),
        forward_batch.out_cache_loc,
    )
```

## 性能声明

MI355X,MiniMax-M2.5 FP8,TP=4,ISL=8192,OSL=1024:输出吞吐 conc=64 时 +2.5%,conc=32 时 +2.3%,conc=4 时最高 +5.9%;conc=128 无回退(+0.4%)。精度:GSM8K 93.3%,与 baseline 持平。

## 验证与回归风险

PR 正文仅给出 GSM8K 精度结果(93.3% 不变),未提到新增单元测试;diff 中无测试文件改动。回归风险集中在本分支的触发条件:需同时满足 aiter 后端、unified (non-SWA) Triton attention 与 FP8 KV cache;该路径下页大小 view 依赖 `self.page_size` 与 head 维度的一致性。其他后端与 NVIDIA GPU 路径不受影响。

## 关联

同系列:[AMD] Optimize MiniMax-M2.5 系列中的 #23611(aiter biased_grouped_topk MoE routing)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
