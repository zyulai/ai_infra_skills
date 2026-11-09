# PR #33568: [Perf] Disable clean_logits in deepgemm fp8_mqa_logits kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-02-06 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 4 个文件(deep_gemm wrapper、sparse_attn_indexer、两个测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/33568 |

## 一句话结论

对 DeepSeek V3.2 sparse attention indexer 关闭 DeepGEMM fp8 MQA logits 的 padding 清洗 kernel,H200 上 e2e 输出吞吐 +15%。

## 动机

作者在 DeepSeek V3.2 的 `sparse_attn_indexer` profiling 中发现,`deep_gemm::sm90_fp8_mqa_logits` 之后总是跟着一个 `deep_gemm::smxx_clean_logits` kernel——因为 vLLM 的 `fp8_mqa_logits`/`fp8_paged_mqa_logits` wrapper 把 `clean_logits` 硬编码为 True。该 kernel 的作用是把 MQA logits 的 padding 值填成 -inf,但它耗时较长;而下游消费方根本不读 padding:`topKPerRowDecode` 只读 `rowEnd` 范围内的行,`topKPerRowPrefill` 同样不读 padding。因此清洗纯属浪费,省掉这次 kernel launch 可直接降延迟。

## 优化点

机制上是一个"删掉多余 kernel launch"的改动:`vllm/utils/deep_gemm.py` 的 `fp8_mqa_logits` 与 `fp8_paged_mqa_logits` 增加 `clean_logits: bool` 参数并透传给 DeepGEMM(后者原来硬编码 `clean_logits=True`);`sparse_attn_indexer.py` 的 prefill(分块路径)与 decode(paged 路径)调用均传 `clean_logits=False`,不再启动 `smxx_clean_logits`。logits 张量中 padding 区域从此保留未定义值,依赖下游 kernel 只在有效行范围内读取这一契约。测试相应把 `clean_logits` 参数化,True 时仍校验 -inf mask 与参考一致,False 时仅对有效区域做 `calc_diff < 1e-3` 的数值比对。

## 关键改动

```text
vllm/utils/deep_gemm.py: fp8_mqa_logits / fp8_paged_mqa_logits 暴露 clean_logits 参数
vllm/model_executor/layers/sparse_attn_indexer.py: prefill 与 decode 调用传 clean_logits=False
tests/kernels/attention/test_deepgemm_attention.py: clean_logits=True/False 双路参数化
tests/kernels/test_top_k_per_row.py: 覆盖 padding 未清洗(非 -inf)时 top-k 行为
```

decode 路径调用(节选):

```python
logits = fp8_paged_mqa_logits(
    ...,
    block_tables,
    schedule_metadata,
    max_model_len=max_model_len,
    clean_logits=False,
)
```

## 性能声明

PR 正文:H200 上 e2e output token throughput 提升 15%。服务 benchmark(deepseek-ai/DeepSeek-V3.2-Exp,TP8、EP、`--max-num-seqs 16`、无 prefix caching,ShareGPT、`--sharegpt-output-len 300`、并发 16、1000 prompts、50 warmup、ignore-eos):输出吞吐 453.60 对 520.05 tok/s,总吞吐 800.48 对 917.74 tok/s,benchmark 时长 661.38 对 576.87 s,Mean TTFT 1359.93 对 433.33 ms,Mean TPOT 30.63 对 29.21 ms。MTP(1 speculative token)配置下收益较小:输出吞吐 493.37 对 510.15 tok/s,TPOT 31.57 对 30.54 ms,acceptance rate 两侧均 38.90%。Profiling 截图显示 main 有 `smxx_clean_logits`、PR 无。

## 验证与回归风险

`pytest tests/kernels/attention/test_deepgemm_attention.py` 与 `tests/kernels/test_top_k_per_row.py` 单元测试通过;两个测试文件均参数化 `clean_logits=True/False`,`test_top_k_per_row` 的参考实现改为逐行仅在 `row_end` 内做 topk(与 kernel 的读取契约一致),并加入 `set_random_seed` 固定随机性。精度验证:GSM8K(lm_eval,DeepSeek-V3.2-Exp)main 0.9560/0.9538 对 PR 0.9568/0.9560;MTP 配置两侧完全一致(0.9591/0.9583)。回归风险:任何直接消费 fp8_mqa_logits 输出且假设 padding 为 -inf 的新调用方会读到未定义值——wrapper 保留 `clean_logits` 参数,需要清洗语义的调用方仍可显式传 True;影响面限于 DeepSeek V3.2 sparse attention indexer 的 SM90/SM100 路径。

## 关联

同区域:DeepSeek V3.2 sparse attention(DeepGEMM fp8_mqa_logits / topKPerRow)路径上的其他优化素材未列出;#21787 同样涉及 `vllm/utils/deep_gemm.py` 的 DeepGEMM 集成面。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
