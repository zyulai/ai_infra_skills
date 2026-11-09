# PR #57407: [ROCm][Perf] Enable layer-aware CSA2 multi-stream overlap for DeepSeek-V4.1-Flash

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-09-27 |
| 分类 | Overlap / Scheduler |
| Label | rocm, ready, deepseek, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/57407 |

## 一句话结论

DSv4.1-Flash ROCm attention 层按层型重排 multi-stream:每步一个 fork/join 替代三个,full CSA2 层模式快 20.76%、reindex 层快 11.90%,低并发 TTFT -32%。

## 动机

DeepSeek-V4.1-Flash 有三种 attention 层:full CSA2 层(2/8/14/20,SWA + compressor + indexer,新建全局 KV cache)、reindex CSA2 层(24/28/32/36,仅 SWA + indexer,对更早层的全局 KV cache 重打分)、其余 dense/pure SWA 层。共享的 v4.1 pipeline 对 full 层每步做 3 个 fork/join(输入投影;Q/SWA-insert vs compressor;indexer vs cache-insert),reindex 层 1 个;ROCm 上每个 fork/join 是每条 side stream 一对 HIP event,multi-stream launch 成本超过 overlap 收益。

## 优化点

按层型定制、每步恰好一个 fork/join 的 ROCm 方案:full CSA2 层在 input norm 后做一次 3-stream fork——default 流跑 fused wqa+wkv → q/kv split+norm → wq_b → fused qnorm/RoPE/SWA-insert,aux0 跑 compressor 链(kv_score GEMM → save/compress/norm → RoPE-quant cache insert),aux1 跑 indexer `weights_proj`;join 后串行执行 indexer K 写(`_produce_k`,其 wk 消费 aux0 latent)与 query 路径(`forward_q`,wq_b 消费 default 流 qr),再进 indexer op 与 sparse MLA。reindex 层输入投影先串行(query 路径消费 qr,overlap 权重投影不值得),随后一个 2-stream fork 让 SWA q 路径与 indexer query 路径并行。dense 层完全串行(`aux_stream_list` 置 None)。新增共享 helper `execute_in_parallel_default_first`(`vllm/utils/multi_stream_utils.py`,`execute_in_parallel` 加 `default_first` 参数):default 链最长,先入队以缩短 step 尾部。fork 的 gate 与 v4.0 多流 gate(#51794/#56853)一致——仅当 fork/join 成为静态 graph 边(capture 内或非 dict metadata)才启用;MRV1 等其余场景回落全串行(#51430)。CUDA 路径除行为中性的重构(indexer query 路径抽为 `DeepseekV4Indexer.forward_q`)外不动。

## 关键改动

```text
vllm/models/deepseek_v41/amd/rocm.py: _forward_csa2_full/_forward_csa2_reindex + ROCm 多流 gate
vllm/models/deepseek_v41/attention.py: 抽出 DeepseekV4Indexer.forward_q
vllm/utils/multi_stream_utils.py: execute_in_parallel 增加 default_first 参数
tests/kernels/test_compressor_kv_cache.py: full/reindex pipeline 流序与 join 语义测试
```

```python
(q, qr, qr_scale, kv), (latent, indexer_weights) = execute_in_parallel(
    default_chain, [compressor_chain, indexer_weights_chain],
    self.ln_events[0], self.ln_events[1:3], aux_streams[:2],
    enable=True, default_first=True)
```

## 性能声明

模式级(trace 对比,conc=4 示例):full CSA2 层新 pattern 比旧 pattern 快 20.76%;reindex 层快 11.90%(后者依据:overlap 收益 4 µs < multi-stream 成本 11 µs,故改为 overlap SWA 与 indexer 过程)。e2e(SA InferenceX `8k1k`,TP4,nightly-rocm100 镜像):conc 1 output tok/s 114.47 → 117.28(+2.45%),mean TTFT 392.61 → 266.06 ms(-32.23%),TPOT -0.84%;conc 4 TTFT -4.76%;conc 16/64 变化 ≤0.46%(作者注明仅 8 层可 overlap,e2e 收益有限)。gsm8k(5-shot,lm_eval):flexible 0.9363 两侧一致,strict 0.9363 → 0.9371。

## 验证与回归风险

新增 ROCm 专属测试:full pipeline 断言执行顺序(wq_b → qnorm_insert → compressor → insert_cache → weights_proj → produce_k → forward_q)、各 stage 落在正确流、join 发布两侧 cache 写;reindex pipeline 断言串行投影 + aux0 上的 forward_q;`forward_q` 抽取的单测(quant tuple 两种返回形态)。风险:ROCm HIP event sync 在非 capture 区域不可靠(event waits 可能 hang),故严格 gate 到 capture/非 dict metadata,任何 gate 判断失误都会走错流;reindex 层输入投影强制串行属行为变化;`_prepare_and_attn_fn` 为 eager 时(MRV1)必须回落串行,否则产生垃圾输出。

## 关联

#51794、#56853(v4.0 multi-stream gate 前序)、#51430(MRV1 wide eager region 约束);与 #60397(DSv4.1 mono decode,本批索引)作用于同一模型的不同路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
