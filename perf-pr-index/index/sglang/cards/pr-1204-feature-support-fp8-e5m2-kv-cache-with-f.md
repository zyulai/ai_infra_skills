# PR #1204: [Feature] Support fp8 e5m2 kv cache with flashinfer

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2024-08-26 |
| 分类 | Quantization(major) |
| Label | feature |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/1204 |

## 一句话结论

新增 `--kv-cache-dtype fp8_e5m2`,KV cache 减半显存占用,llama2-13b A100 吞吐 +17.8%。

## 动机

此前 KV cache 只能以模型 dtype(FP16/BF16)存储,长上下文与大 batch 下的 KV 显存占用是吞吐瓶颈。以 FP8 e5m2 存储 KV 可将每 token 的 cache 体积减半,同显存下容纳更多 token;但需要 attention 后端(FlashInfer)感知 cache dtype 才能正确计算。该 PR 让 KV cache dtype 与模型 dtype 解耦。

## 优化点

新增 `--kv-cache-dtype` 参数(可选 auto/fp8_e5m2),`model_runner` 在 `init_memory_pool` 时据此解析 `kv_cache_dtype`(fp8_e5m2 仅在未禁用 FlashInfer 且非 MLA 时生效,否则回退 auto 并告警),`profile_max_num_token` 改用 `kv_cache_dtype` 计算 cell size,使 KV 容量按 FP8 计算。内存池侧引入 store/view 双 dtype 机制:因 PyTorch 的 Tensor index_put 不支持 `torch.float8_e5m2`,buffer 以 `torch.uint8` 物理存储,读取时 `.view(self.dtype)` 还原;写入走新抽象方法 `set_kv_buffer`,在池内统一做 dtype 转换与视图写入。`radix_attention.store_kv_cache` 从直接索引写 buffer 改为调用 `set_kv_buffer`;`forward_batch_info` 的 FlashInfer wrapper 传入 `data_type=model_runner.kv_cache_dtype` 与 `q_data_type`,让 kernel 按 FP8 KV 计算。

## 关键改动

```text
python/sglang/srt/mem_cache/memory_pool.py: BaseTokenToKVPool 抽象化,新增 store_dtype(uint8)/set_kv_buffer,fp8 存储 + view 读取
python/sglang/srt/model_executor/model_runner.py: 解析 kv_cache_dtype;KV 容量估算与池初始化改用 kv_cache_dtype
python/sglang/srt/model_executor/forward_batch_info.py: FlashInfer wrapper 传 data_type/q_data_type
python/sglang/srt/server_args.py: 新增 --kv-cache-dtype CLI 参数
python/sglang/srt/layers/radix_attention.py: store_kv_cache 改走 token_to_kv_pool.set_kv_buffer
```

```python
# memory_pool.py
if dtype == torch.float8_e5m2:
    # NOTE: Store as torch.uint8 because Tensor index_put is not
    # implemented for torch.float8_e5m2
    self.store_dtype = torch.uint8
else:
    self.store_dtype = dtype
...
def set_kv_buffer(self, layer_id, loc, cache_k, cache_v):
    if cache_k.dtype != self.dtype:
        cache_k = cache_k.to(self.dtype)
    ...
    if self.store_dtype != self.dtype:
        self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
```

## 性能声明

PR 正文:llama2-13b-chat on A100,`--disable-radix-cache --tp=1`,bench_serving ShareGPT 1000 prompts(request-rate 128):吞吐 7.04 req/s → 8.30 req/s(+17.8%);MMLU(nsub=10)平均精度 0.487 → 0.488,gsm8k 0.340 → 0.334,即"无精度退化"。PR 同时注明该收益是模型相关的:llama3-8b 也测过,但没有性能提升。

## 验证与回归风险

PR 正文未列出新增单元测试,验证以端到端 benchmark + MMLU/gsm8k 精度评测为主。风险路径:fp8_e5m2 仅支持 FlashInfer attention 路径,Triton kernel 与 MLA 被显式回退/排除;`set_kv_buffer` 的逐次 dtype 转换在写入路径引入额外开销;抽象接口变更要求所有 TokenToKVPool 子类实现 `set_kv_buffer`/`get_kv_buffer`。

## 关联

无(pack 素材未提及其他同分类或前序/后续 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
