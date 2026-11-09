# PR #63669: [ROCm][Perf] Make MXFP4 the default indexer K cache dtype for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-10-26 |
| 分类 | Quantization (tier: major) |
| Label | rocm, deepseek, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 2 个文件(attention 配置与 ROCm 平台) |
| 链接 | https://github.com/vllm-project/vllm/pull/63669 |

## 一句话结论

把 DeepSeek-V4 在 ROCm CDNA4 的 indexer K cache 默认改为 MXFP4,TTFT 最多降 30.7%、KV 容量 +9.2%。

## 动机

PR 正文 Overview:#60834 为 ROCm gfx950 上的 DeepSeek-V4 带来了分页 MXFP4 稀疏 indexer(dense path),但需显式 `--attention-config '{"indexer_kv_dtype": "mxfp4"}'` 开启。其测量显示 MXFP4 indexer 在几乎所有维度优于 FP8:更低 TTFT(最多 -30.7%)与 TPOT、+9.2% KV cache 容量(每压缩 token 68 B vs 132 B)、精度近无损(GSM8K -0.16pt、GPQA Diamond 维持 0.7121)。因此本 PR 把 MXFP4 设为默认并打印日志。

## 优化点

在 ROCm 平台的 `check_and_update_config` 中新增 `_maybe_default_dsv4_mxfp4_indexer`:仅当 `hf_config.model_type == "deepseek_v4"`(V4-Pro / V4-Flash)、`indexer_kv_dtype` 仍为 `"auto"`、未开 `indexer_sparse_logits`、`get_cdna_version() == 4`、未启用 DCP/PCP(`decode_context_parallel_size`、`prefill_context_parallel_size` 均为 1)且 `rocm_aiter_ops.is_enabled()` 时,把 `indexer_kv_dtype` 改写为 `"mxfp4"` 并用 `logger.info_once` 说明如何回退到 FP8。只重写 `"auto"`,MXFP4 路径跑不起来时静默回退 FP8,保证原本可用的配置不会失败。attention 配置里同步更新 `indexer_kv_dtype` 的注释。正文明确其他平台与模型不变:CUDA、XPU、CPU 行为照旧,DeepSeek-V4.1 仍保持 opt-in。

## 关键改动

```text
vllm/platforms/rocm.py: 新增 _maybe_default_dsv4_mxfp4_indexer,在 CDNA4 + aiter 且 model_type=deepseek_v4 时把 auto 改为 mxfp4,由 check_and_update_config 调用
vllm/config/attention.py: 更新 indexer_kv_dtype 注释,说明 "auto" 对 DeepSeek-V4 在 ROCm CDNA4 + aiter 下选 mxfp4
```

关键机制代码(默认值改写与守卫条件):

```python
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) != "deepseek_v4":
        return
    ...
    if (attention_config.indexer_kv_dtype != "auto"
            or attention_config.indexer_sparse_logits
            or get_cdna_version() != 4
            or parallel_config.decode_context_parallel_size > 1
            or parallel_config.prefill_context_parallel_size > 1):
        return
    if not rocm_aiter_ops.is_enabled():
        return
    attention_config.indexer_kv_dtype = "mxfp4"
```

## 性能声明

正文数据全部引自 #60834(DeepSeek-V4-Pro)。固定序列长度 8k1k、TP8、MTP=2:conc=1 输出吞吐 145.99 → 156.34 tok/s(+7.09%)、mean TTFT 597.24 → 413.91 ms(−30.70%)、mean TPOT 6.24 → 5.96 ms(−4.49%);conc=4 吞吐 +2.64%、TTFT −13.25%;conc=16 吞吐 +0.92%、TTFT −8.46%;conc=64 吞吐 +1.64%、TTFT −0.90%。Agentic replay(DP8、MTP=3、prefix caching):mean TPOT 46.01 → 43.42 ms(−5.62%)、mean TTFT 2442.24 → 2321.05 ms(−4.96%)、KV cache 容量 4,589,684 → 5,010,749 tokens(+9.17%)、输出吞吐 697.25 → 675.83 tok/s(−3.07%,正文称在噪声内)。精度(同引自 #60834):GSM8K 0.9492 → 0.9477、GPQA Diamond 均为 0.7121。

## 验证与回归风险

正文 Validation 直接采用 #60834 的 benchmark 与 accuracy 结果,未在本 PR 内新增测试;正文提示 TTFT 在同配置不同运行间波动可达 13%,其增益宜视为指示性。影响路径:仅 ROCm CDNA4(MI350/MI355)上的 DeepSeek-V4(V4-Pro / V4-Flash)默认 indexer dtype;CUDA、XPU、CPU、DeepSeek-V4.1 与其他模型保持不变。MXFP4 相对 FP8 的精度变化(GSM8K 约 -0.16pt)及 MXFP4 路径本身可用性是需关注的回归点,代码通过「只改 auto + 静默回退 FP8」降低风险。

## 关联

前序 PR #60834(正文所引 MXFP4 稀疏 indexer 的实现与全部 benchmark/accuracy 数据来源)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
