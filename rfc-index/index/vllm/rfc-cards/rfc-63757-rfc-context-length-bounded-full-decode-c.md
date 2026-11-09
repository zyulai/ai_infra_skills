# RFC #63757: [RFC]: Context-length-bounded FULL decode CUDA graphs in Model Runner V2

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63757](https://github.com/vllm-project/vllm/issues/63757) |
| 作者 | eppaneamd |
| 创建日期 | 2026-10-26 |
| 状态 | 🟢 open |
| labels | quantization, rocm |
| 热度 | 👍 0 · 💬 1 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议按更小的 max_seq_len 上界捕获 FULL decode CUDA graph 副本并按步分派,短上下文时跳过索引器等死工作,不破坏 replay 正确性。

## 提案动机
Model Runner V2 为每张 FULL decode CUDA graph 以 `max_seq_len = max_model_len` 捕获 attention metadata,图对任意 replay 上下文有效;捕获期基于 `max_seq_len` 的 host 侧决策因此都按最长上下文做出并固化进图。部分模型在步内所有上下文都短时可跳过的工作因此无法跳过:GLM-5-Next 的 sparse indexer 在所有上下文都装进 `index_topk`(GLM-5.3-Flash 上为 2048)时,top-k 就是全部因果 token,q projection、head-weight projection、q 量化、paged-MQA logits 与 top-k 全是死工作;DeepSeek-V4 / V4.1 的 indexer 在 eager 步已会跳过,但用 `not torch.cuda.is_current_stream_capturing()` 守卫(#52492),FULL decode graphs 永远走不到该捷径。#52492 与 #59709 确立的规则——不允许 replay 可使其失效的捕获期捷径——是对的;缺的是一张捕获时 `max_seq_len` 更小、且仍能约束其所有 replay 的 FULL graph。

## 提出的方案
- **Declare**:模块声明 `cudagraph_seq_len_bound: int | None`;runner 从 `model.modules()` 收集低于 `max_model_len` 的去重 bound 值。
- **Capture**:每张 FULL decode graph 按 bound 各捕获一份副本,所有 backend 的 metadata 中 `max_seq_len = bound`。
- **Dispatch**:`BatchExecutionDescriptor` 增加 `max_seq_len`,像 `max_query_len` 一样匹配;上下文全部装入某个 bound 的 decode 步 replay 该副本(tightest first),其余步 replay 默认图;bound 来自 `prepare_inputs` 已在用的 CPU 数组,无 device 读。
- **模块规则**:只跳过 `max_seq_len <= bound` 使其成为死工作的工作,依据模块收到的 metadata 判断,且仅在 FULL graph 捕获时;不在 PIECEWISE 捕获中(其跑在 dummy batch metadata 上),也不在 eager 步(含每次捕获前的 warm-up——否则冷启动可能在捕获内 compile/autotune 而失败,参考实现首 commit 在 breakable `FULL_AND_PIECEWISE` 下出现过)。
- 不变量:FULL graph 捕获时的 `max_seq_len` 是其 replay 的每个 batch、每个上下文的上界(默认图为 `max_model_len`,副本为 bound)。
- Scope:PCP/DCP 下关闭;opt-out 经 `CompilationConfig.cudagraph_specialize_seq_len`;PIECEWISE-only 部署、混合 FULL 图与 micro-batched(DBO)图不获副本;跨过 bound 的步切换到默认图。

## 优化点与收益
正文照抄的数字,条件为 GLM-5.3-Flash MXFP4、2× MI355X、TP2、one image per request(参考实现 #63758,含机制与 GLM-5-Next indexer):
- `FULL_DECODE_ONLY`:并发 16-64 时 output throughput +2.32%(SE 0.44 pp),并发 4 时 +12.3%;跨过或超出 bound 的 text rows 在噪声内;GSM8K 不变。代价:每 capture size × bound 一张额外图——graph memory 0.79 → 1.07 GiB per rank,KV cache −0.03%。
- Breakable `FULL_AND_PIECEWISE`:并发 16-64 时 +4.50%(SE 1.92 pp)。
- 第二个用户(follow-up PR,叠在参考实现上):DeepSeek-V4 / V4.1(V4.1-Flash 由 compress ratios 1 与 2 得 bounds 512 与 1025)。
- 正文注明未测:NVIDIA、TP 1/4/8、DP > 1。

## 现状与落地
🟢 open(2026-10-26 创建),labels 为 quantization、rocm。Feedback period 至少一周,期间参考实现保持 draft。待决问题:与 #62749 记录的 FULL decode metadata build keying(按 batch shape、`max_seq_len = max_model_len`)如何协调;DP 下 bound 是 per rank 还是做 max-over-ranks(后者会让一个长上下文 rank 禁用所有 rank 的副本;DP > 1 未测);默认开启还是 opt-in(实现里默认开,类比 `cudagraph_specialize_lora`)。正文还注明:检索 open PRs 未发现按上下文长度 keying decode graphs 的先例;Model Runner V1 图不受影响。

## 关联
- 捷径守卫出处: https://github.com/vllm-project/vllm/issues/52492
- 规则出处: https://github.com/vllm-project/vllm/issues/59709
- 参考实现: https://github.com/vllm-project/vllm/issues/63758
- metadata build 冲突: https://github.com/vllm-project/vllm/issues/62749
- Prior art:SGLang `python/sglang/srt/layers/attention/graph_variants.py`(按 `max_kv_len <= index_topk` 选 dense/sparse DSA decode graph)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
