# PR #21190: [Whisper] Enable CUDA graph support and timestamp for whisper model

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-03-28 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | blackwell, run-ci |
| Diff | diffstat 缺失(pack 未记录);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/21190 |

## 一句话结论

把 Whisper 的手写 bmm+mask cross-attention 换成 RadixAttention 路径并修好图捕获,让 encoder-decoder 模型首次可用 CUDA graph:吞吐 +36%、延迟 -27%~-29%,WER 不变。

## 动机

Whisper 此前用自定义 `bmm + mask` cross-attention 实现,配合 Python 侧 `_encoder_cache` dict 缓存 encoder 输出,这与 CUDA graph capture/replay 完全不兼容——`server_args.py` 里甚至直接对所有 encoder-decoder 模型强制 `disable_cuda_graph = True`。整个 decode 阶段因此吃不到图回放的低 launch 开销收益。旧实现里还有大量 per-request Python 循环(逐请求构造 block-diagonal mask、按 req_pool_indices 查 dict),是 decode 路径上的明显瓶颈。

## 优化点

机制上的核心是把 cross-attention 收编进标准 KV cache 体系:(1) `WhisperAttention`(cross-attention 分支)删除整段手写 bmm/softmax/masked_fill,改为 prefill 时把 encoder KV 投影后经 `encoder_out_cache_loc` 存入 KV pool,decode 时 `k=None, v=None` 走 `RadixAttention` 从 pool 读缓存 KV;(2) 删除 `_encoder_cache` dict,encoder 输出改经 scheduler/attention backend 的原生 encoder-decoder cache 流转,`pad_input_ids` 前置 `max_source_positions`(1500)个 dummy encoder token 以便正确分配 encoder KV 槽位;(3) CUDA graph 捕获修复:`encoder_len_fill_value` 从 0 改为 `max_source_positions`,否则捕获时 cross-attention 被跳过(之前填 0 导致图里没有 cross-attention kernel);(4) 修 FlashInfer decode cross-attention planning:`update_cross_attention` 之前把 decoder `seq_lens_cpu` 错传给 cross-attention wrapper,导致 `global_override_indptr_cpu` 把 KV 长度从 1500 覆盖到约 5,现改为用 `encoder_lens`(并缓存 CPU 拷贝避免每次 GPU->CPU 传输);(5) 服务端自动配置:Whisper 架构自动选 flashinfer backend(trtllm_mha/fa3 缺 `encoder_out_cache_loc` 支持)、自动禁 radix cache(encoder token padding 与 prefix caching 冲突);(6) 附带 OpenAI 兼容 `verbose_json` 时间戳输出:decoder 起始 token 用 `<|0.00|>` 替代 `<|notimestamps|>`,serving 侧按 timestamp token(基 id 50365,每步 0.02s)切分 segments;(7) `position_ids` clamp 到 `max_target_positions - 1` 防 warmup 越界 448 项位置表。

## 关键改动

```text
python/sglang/srt/models/whisper.py: 删手写 bmm cross-attention 与 _encoder_cache,走 RadixAttention + KV pool
python/sglang/srt/model_executor/cuda_graph_runner.py: encoder_len_fill_value 取 max_source_positions,保证 cross-attention 入图
python/sglang/srt/layers/attention/flashinfer_backend.py: update_cross_attention 改用 encoder_lens 做 cross-attention plan
python/sglang/srt/server_args.py: Whisper 自动选 flashinfer backend、禁 radix cache,删除全局 encoder-decoder 禁图
python/sglang/srt/entrypoints/openai/serving_transcription.py: verbose_json segments 解析(timestamp token -> start/end)
```

cross-attention 新路径:

```python
if self.is_cross_attention:
    q, _ = self.q_proj(hidden_states)
    q = q * self.scaling
    if cross_hidden_states is not None:
        kv, _ = self.kv_proj(cross_hidden_states)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
    else:
        k = None
        v = None
    attn_output = self.attn(q, k, v, forward_batch)
```

## 性能声明

条件:`openai/whisper-large-v3`,数据集 `D4nt3/esb-datasets-earnings22-validation-tiny-filtered`(511 samples),concurrency=1,`benchmark/asr/bench_sglang.py`。No CUDA Graph -> CUDA Graph:WER 12.77% -> 12.77%(identical);Avg Latency 0.406s -> 0.297s(-27%);Median Latency 0.376s -> 0.267s(-29%);P95 Latency 0.837s -> 0.601s(-28%);Throughput 2.40 req/s -> 3.26 req/s(+36%);Token Throughput 44.71 tok/s -> 60.72 tok/s(+36%)。PR 未注明测试硬件型号(label 含 blackwell,但正文未给出具体 GPU)。

## 验证与回归风险

新增 `test/manual/test_whisper_cuda_graph.py`:验证 CUDA graph 开启下基本转录、多次请求输出一致(图回放一致性)、转录质量关键词;并有 `--disable-cuda-graph` 的对照组用例。精度由 WER identical(12.7690%,earnings22 511 samples)背书。回归风险:encoder-decoder 通用的 `encoder_len_fill_value`/`_dummy_run` 改动影响所有 encoder-decoder 模型(mllama 等仍可用 radix cache,只有 Whisper 自动禁);flashinfer backend 的 cross-attention planning 修正是所有用该路径模型的共享代码;Whisper 用户若依赖 radix cache 行为会有变化(自动禁用)。

## 关联

相关:#21161(正文 Related)。同区域:encoder-decoder CUDA graph 支持与 flashinfer cross-attention 相关 PR 编号未在正文注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
