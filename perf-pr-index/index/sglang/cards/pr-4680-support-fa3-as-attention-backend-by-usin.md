# PR #4680: Support FA3 as Attention backend by using `--attention-backend fa3`

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hebiao064 |
| 合并 | 2025-03-24 |
| 分类 | Speculative Decoding (tier: major) |
| Label | high priority, performance |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/4680 |

## 一句话结论

新增 FA3 attention backend(`--attention-backend fa3`):decode 吞吐略高于 FlashInfer 基线,大 batch/长输入不 OOM,Gemma-2 场景精度显著更好。

## 动机

SGLang 当时只有 FlashInfer/Triton/torch_native 三个 attention backend。FlashInfer 在大 batch、大输入时会 OOM 而 FA3 不会;作者希望接入 FA3 的 `flash_attn_with_kvcache` 作为第四个 backend(Roadmap 见 issue #4709)。PR 正文给出定性结论:prefill 吞吐与基线持平,decode 吞吐略高,精度略优。

## 优化点

新建 `FlashAttentionBackend`(约 295 行),extend 与 decode 统一走 FA3 的 `flash_attn_with_kvcache`,paged KV cache 经 `page_table`/`cache_seqlens` 传入,支持 `softmax_scale`、`window_size`(sliding window,双侧包含)、`softcap` 与 k/v descale。核心机制是 `init_forward_metadata` 预计算 `cu_seqlens_q/k`、`cache_seqlens_int32`、`max_seq_len_k` 与 page table,避免 forward 内重复计算;decode 的 CUDA graph 提供 capture/replay 支持(replay 时重算 metadata 并按 `max_seq_len_k` 局部刷新 page table)。注册时强制 SM>=90(Hopper 及以上),低于则拒绝启动。本 PR 为 Beta:multimodal、page size > 1、FP8、MLA、speculative decoding 均不支持,列为后续工作。

## 关键改动

```text
python/sglang/srt/layers/attention/flashattention_backend.py: FA3 backend(extend/decode + CUDA graph metadata,新文件)
python/sglang/srt/model_executor/model_runner.py: 注册 "fa3" backend,SM>=90 检查与 Beta 警告
python/sglang/srt/server_args.py: --attention-backend choices 增加 "fa3"
python/sglang/test/attention/test_flashattn_backend.py: extend/decode/prefix 场景单元测试(新文件)
```

## 性能声明

PR 正文未给出量化吞吐数字(latency benchmark 为聚合截图与外部 sheet,正文文字结论如下):

- Prefill 吞吐与当时基线持平;decode 吞吐略高于基线;精度略优。
- FlashInfer 在 batch size、输入长度大时会 OOM,FA3 不会。
- GSM8K(8-shot,1319 题,`benchmark/gsm8k/bench_sglang.py`):Meta-Llama-3.1-8B-Instruct FA3 0.793 vs FlashInfer 0.789;Qwen2.5-7B-Instruct 0.823 vs 0.789;Gemma-2-9B 0.724(Torch Native 0.730)vs FlashInfer 0.132(正文标注 potential bug)。硬件未注明。

## 验证与回归风险

新增单元测试 `test_flashattn_backend.py` 覆盖 forward_extend、forward_decode 与带 prefix 的 extend,断言输出形状/dtype/无 NaN(page_size=1)。sliding window 用 Gemma-2 验证。风险面:仅支持 page size 1、MHA 模型(Llama/Qwen/Gemma 类)、无 FP8/MLA/multimodal/speculative decoding;prefill 模式不支持 CUDA graph capture(显式 raise);requiring SM>=90。文档与 benchmark 交付项在合并时未勾选。

## 关联

Roadmap issue #4709(FA3 接入计划);后续 TODO(Page Size > 1、multimodal、speculative decoding、MLA、FP8 支持均为本 PR 之后的工作);pack 内未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
