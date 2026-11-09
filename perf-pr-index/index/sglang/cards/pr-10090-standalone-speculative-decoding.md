# PR #10090: Standalone speculative decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Qiaolin-Yu |
| 合并 | 2025-09-08 |
| 分类 | Speculative Decoding |
| Label | enhancement, high priority, speculative-decoding |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/10090 |

## 一句话结论

新增 STANDALONE 投机解码:用独立小模型(如 Llama-3.2-1B)为 Llama-3.1-70B 起草,单并发 decode 吞吐约 2.7 倍。

## 动机

PR 之前 SGLang 的投机解码只支持 EAGLE/EAGLE3/NEXTN,依赖训练好的 EAGLE head,无法直接用任意现成小模型(如 Llama-3.2-1B-Instruct)作为 draft model。本 PR 关闭 issue #9785,让用户可以指定一个完全独立的小模型来起草,补齐投机解码的模型选型自由度。

## 优化点

在 `SpeculativeAlgorithm` 中新增 `STANDALONE` 枚举,并新增 `StandaloneWorker`(继承 `EAGLEWorker`)作为 draft worker:它复用 target worker 的 req_to_token_pool 与 token_to_kv_pool_allocator,但 draft 与 target 各自持有独立 KV cache pool;draft 模型的 context_length 被强制覆盖为与 target 一致。draft forward 复用 EAGLE 的多步起草/验证流水线,因此 CUDA graph、draft-extend 等 EAGLE 基础设施全部沿用,只是在 `cuda_graph_runner`、`schedule_batch` 等处把 `is_eagle()` 判断扩展为 `is_eagle() or is_standalone()`。server_args 层面为 STANDALONE 预留更大显存(6GB vs 其他算法 2GB,因 draft 模型更大)、默认参数取 (3, 1, 4)、显式 assert 不支持 dp attention。另含一个针对 gpt-oss draft 的 `out_cache_loc.contiguous()` 修复(gpt-oss rope kernel 要求 cache_loc 连续)。

## 关键改动

```text
python/sglang/srt/speculative/standalone_worker.py: 新增 StandaloneWorker,复用 EAGLEWorker 起草逻辑与 target 的内存池
python/sglang/srt/speculative/spec_info.py: SpeculativeAlgorithm 增加 STANDALONE 与 is_standalone()
python/sglang/srt/managers/scheduler.py: spec_algorithm 为 STANDALONE 时实例化 StandaloneWorker
python/sglang/srt/server_args.py: CLI 增加 STANDALONE 选择、显存预留 6GB、默认 (3,1,4)
python/sglang/srt/model_executor/cuda_graph_runner.py: standalone 复用 EAGLE 的 cuda graph spec_info 路径
```

## 性能声明

- 测试条件:meta-llama/Llama-3.1-70B-Instruct(target,TP4,mem-fraction-static 0.7)+ Llama-3.2-1B-Instruct(draft);`bench_serving` random 数据集,input/output 各 1024 token,range-ratio 0.98,5 prompts,max-concurrency 1。
- Without spec:63.87 token/s(日志显示 63.66 token/s)。
- With standalone spec:173.38 token/s(日志行显示 181.92 token/s,accept len 3.85)。
- 精度:GSM8K 8-shot、1319 题:Accuracy 0.941,Invalid 0.000,Latency 60.287s。

## 验证与回归风险

新增 `test/srt/test_standalone_speculative_decoding.py` 并注册进 run_suite.py(权重 250),覆盖 fa3/triton/flashinfer 三种 attention backend,断言 GSM8K 准确率 > 0.7、平均接受长度 > 3.6。风险点:STANDALONE 不支持 dp attention(assert 拦截);与 EAGLE 共用路径被改动——`eagle_draft_extend_cuda_graph_runner.replay` 中 hidden_states 拷贝新增 shape 相等判断、decode batch 准备逻辑扩展到 standalone,这些改动影响所有投机解码用户;gpt-oss draft 的 contiguous 修复只针对 STANDALONE+GptOssForCausalLM 组合。

## 关联

同分类:PR #28782(Spec: FlashInfer CUDA graph for EAGLE draft-extend)。前序 issue:#9785。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
