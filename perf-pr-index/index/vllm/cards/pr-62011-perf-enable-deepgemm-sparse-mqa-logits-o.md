# PR #62011: [Perf] Enable DeepGEMM sparse MQA logits on the FP8 indexer cache, DSv4.1

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-10-27 |
| 分类 | Quantization (tier: major) |
| Label | ready, ci/build, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 8 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/62011 |

## 一句话结论

让 DeepGEMM sparse MQA logits 支持默认 FP8 indexer cache,4×B300 DSv4.1 长上下文 decode 吞吐最高 +11%。

## 动机

`attention_config.indexer_sparse_logits`(#56254)让 DeepSeek-V4.1 的 candidate-consuming indexer layers 只用 DeepGEMM sparse MQA-logits kernel 给 candidate blocks 打分,而非对整个 context 算 dense logits。但该 flag 此前要求 MXFP4 indexer cache(`indexer_kv_dtype="mxfp4"`),默认 FP8 cache 下启动直接报 `ValueError: attention_config.indexer_sparse_logits requires the MXFP4 indexer cache`。限制源自 #61149 之前:FP8 cache 每 token 存一个 fp32 scale,而 sparse kernel 要 UE8M0-packed(granularity-32)scales。

## 优化点

#61149 之后,SM100 的 FP8 indexer cache 把每个 token 的 power-of-two scale 存成 4 个相等的 UE8M0 bytes,正是 DeepGEMM 期待的 MXFP8 布局;DeepGEMM 的 sparse kernel 也已有 FP8(`float_e4m3_t`、`SM100_MMA_MXF8F6F4_SS`)variant,覆盖 contiguous(prefill)与 paged(decode)两条路径,其 paged accessor 读的页布局与 vLLM FP8 cache 相同(`PAGE_KV x 128` value bytes + `PAGE_KV` uint32 scales)。PR 把 FP8 cache 接进已有 sparse path,使 decode 加速不再需要把 K cache 降到 4 bit。测试侧新增 `_quant_k` / `_quant_q` 按 `kv_dtype` 构造 MXFP4 或 FP8 的 Q/K 与 packed UE8M0 scales,并把 sparse logits 用例参数化为 `["mxfp4","fp8"]`;FP8 Q 用 unit UE8M0 scales(scale 折进权重)。

## 关键改动

```text
vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py, vllm/model_executor/layers/sparse_mqa_indexer.py, vllm/v1/attention/backends/mla/sparse_indexer.py: 让 sparse MQA logits 路径接受 FP8 indexer cache
vllm/config/attention.py, vllm/model_executor/models/config.py, vllm/models/deepseek_v41/attention.py: 放开 indexer_sparse_logits 对 mxfp4 的校验并接入 DSv4.1
tests/model_executor/layers/test_mla_short_prefill_indexer.py: 新增 _quant_k/_quant_q,kv_dtype 参数化 mxfp4/fp8,校验 sparse 与 dense masked logits 一致
.buildkite/test_areas/kernels.yaml: kernels CI(SM100 only)加入 sparse_mqa_logits 用例与新测试文件
```

关键代码(FP8 K 的 UE8M0 packed scale 构造):

```python
    amax = x.float().abs().amax(dim=-1).clamp(min=1e-4)
    exp = torch.ceil(torch.log2(amax / 448.0)).to(torch.int32)
    scale = torch.exp2(exp.float())
    values = (x.float() / scale[:, None]).to(torch.float8_e4m3fn)
    ue8m0 = ((exp + 127).to(torch.int64) * 0x01010101).to(torch.int32)
```

## 性能声明

条件:`DeepSeek-V4.1-Flash`、TP4、4x B300、`--kv-cache-dtype fp8 --no-enable-prefix-caching`、greedy decoding;decode 与 prefill throughput 由 wall time 对 output length(256/1024/2048 output tokens)线性拟合得到,每点跑两次。decode tok/s(MAIN → PR)/speedup:64 x 8K 6,459 → 6,584(1.02x)、32 x 100K 3,920 → 4,117(1.05x)、16 x 256K 2,293 → 2,535(1.11x)、8 x 512K 1,301 → 1,399(1.08x)、4 x 1M 692 → 745(1.08x);prefill tok/s 对应 124,569 → 126,088(1.01x)、117,066 → 117,774(1.01x)、103,739 → 104,733(1.01x)、85,609 → 87,355(1.02x)、62,604 → 63,967(1.02x)。正文称 decode 每点都改善、最高 +11% higher decode throughput;prefill 基本持平,因 #58132 的 decoder SWA bounded replay 在 prefill 只对每个请求最后 128 tokens 跑 candidate-consuming layers,sparse path 几乎没有 prefill indexer 工作可省。Serving matrix(1024 output tokens、每点 4 x concurrency prompts):Median TPOT 在 29/29 点更低(1.01-1.10x,中位 1.03x)、output throughput 在 28/29 点更高(中位 1.02x),并发 1 的 prefill-only TTFT 在 4K 到 1M 持平(0.99-1.04x),4K/16K 短上下文 TPOT 1.01-1.06x 无回退。Accuracy(gsm8k):MAIN 0.901 / 0.000 invalid / 3761.825 output tok/s;PR 0.908 / 0.002 invalid / 3878.109 output tok/s。

## 验证与回归风险

CI 侧 `.buildkite/test_areas/kernels.yaml` 加入 `pytest -v -s model_executor/layers/test_mla_short_prefill_indexer.py -k sparse_mqa_logits`(注释标注 SM100 only、MXFP4 与 FP8 indexer cache);`_skip_unless_sm100_sparse_kernels` 要求 `torch.cuda.get_device_capability()[0] == 10` 且 DeepGEMM >= 2.8(导入改为 `vllm.third_party.deep_gemm`)。测试校验 sparse-MQA candidate 路径与 `torch.topk` over dense bf16 logits(掩码到 candidates)在候选列上逐位一致,含 CUDA graph capture。风险面:改动 `config/attention.py` 的启动校验与 DSA sparse MQA logits / sparse indexer 共享路径,涉及 FP8 与 MXFP4 两种 indexer cache、SM100 限定;正文未列 gsm8k 之外的精度评测,也未列非 SM100 硬件的验证。

## 关联

PR 正文明确:#56254 引入 `attention_config.indexer_sparse_logits`;#61149 使 SM100 FP8 indexer cache 改为 UE8M0-packed scale 布局;#58132(decoder SWA bounded replay)解释 prefill 持平原因。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
