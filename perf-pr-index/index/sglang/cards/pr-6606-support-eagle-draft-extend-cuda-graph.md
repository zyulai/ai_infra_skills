# PR #6606: Support EAGLE draft extend CUDA graph

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2025-05-27 |
| 分类 | Speculative Decoding |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 + / - 数字),diff 含 5 个文件,其中 1 个新文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/6606 |

## 一句话结论

为 EAGLE 投机解码的 draft extend 阶段补上 CUDA graph,DSV3 bs=1 per_user_throughput +11%。

## 动机

EAGLE 投机解码流程里,draft 模型的 decode 阶段已有 CUDA graph 支持,但 decode 之后做 draft extend(`prepare_extend_after_decode` 后的 draft forward)仍是逐次 eager 执行。该阶段每个 verify 后都要跑一次,CPU 启动开销反复出现,是投机解码小 batch 场景下 per-user 吞吐的直接损失来源。PR 之前 `init_cuda_graphs` 中 draft extend 分支直接 `raise NotImplementedError()`。

## 优化点

新增 `EAGLEDraftExtendCudaGraphRunner`,把 `ForwardMode.DRAFT_EXTEND` 的 draft 模型 forward 整段捕获为 CUDA graph,按 batch size 捕获/回放,与既有 draft decode graph runner 同一套全局 memory pool。由于 CUDA graph 要求静态形状,`prepare_extend_after_decode` 增加 `pad_input`:将 input_ids、hidden_states、positions、out_cache_loc 补零 padding 到 `batch_size * (speculative_num_steps + 1)` 的静态长度,回放时再按真实 bs 截断输出。FA3 attention backend 侧新增 draft extend 专用的 graph metadata(cache_seqlens、cu_seqlens_q/k、page_table 等),capture 时静态分配、replay 时用 cumsum(accept_length / cache_seqlens) 原地刷新,page_table 由 `req_to_token` 按 page 重建。当前仅支持 FA3 backend,其他 backend 留待后续 PR。

## 关键改动

```text
python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py: 新文件,EAGLEDraftExtendCudaGraphRunner,捕获/replay draft extend forward
python/sglang/srt/layers/attention/flashattention_backend.py: FA3 增加 draft_extend_metadata 的 init/capture/replay 逻辑
python/sglang/srt/speculative/eagle_utils.py: prepare_extend_after_decode 增加 pad_input,补齐静态长度
python/sglang/srt/speculative/eagle_worker.py: 接入 runner,fa3 时创建 draft_extend_attn_backend,forward_draft_extend_after_decode 走 graph
python/sglang/srt/server_args.py: 投机解码时显存预留从 18GB 提到 20GB
```

关键机制(pad_input 使形状静态化):

```python
static_len = speculative_num_steps + 1
padded_input_size = batch_size * static_len
padded_len = padded_input_size - batch.input_ids.shape[0]
if padded_len > 0:
    new_input_ids = torch.nn.functional.pad(batch.input_ids, (0, padded_len), value=0)
    ...
    padded_cache_loc = torch.zeros(padded_len, dtype=batch.out_cache_loc.dtype, ...)
```

## 性能声明

PR 正文 benchmark(bench_serving,random 输入/输出各 1000 token):

- DSV3(DeepSeek-V3-0324,--tp 8,EAGLE,num_steps=3、topk=1、draft_tokens=4):per_user_throughput bs=1 提升 **11%**(123.947 → 139.708 tok/s),bs=32 提升 **5%**(51.893 → 54.394 tok/s)。
- Llama-3-8B-Instruct(EAGLE draft model lmsys/sglang-EAGLE-LLaMA3-Instruct-8B,FA3,float16,num_steps=2、draft_tokens=3):per_user_throughput bs=1 210.866 → 221.192,bs=16 148.249 → 157.430,bs=32 134.800 → 139.793;各并发档 TPOT 均略有下降。

## 验证与回归风险

PR 未注明新增单元测试,验证依赖 bench_serving 端到端对比与 profile 截图。风险点:padding 逻辑只服务于 graph 路径,`can_run` 不满足时回退 eager 路径仍需正确初始化 metadata;draft extend graph 额外占用显存,server_args 相应提高预留(20GB),显存紧张配置可能 capture 失败(PR 中给出了降低 mem-fraction-static / cuda-graph-max-bs 的提示);其他 attention backend 的 draft extend 仍是 `NotImplementedError` 路径。

## 关联

同区域:EAGLE draft decode CUDA graph(已有实现,本 PR 复用其 capture/replay 框架);同分类投机解码相关 PR #6109(FlashMLA + MTP)、#9873、#24663。Co-authored-by: @kssteven418。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
