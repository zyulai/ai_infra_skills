# PR #58472: [KV Connector] Fix DecodeBench fp8 fill values and add a startup fill mode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @liuzijing2014 |
| 合并 | 2026-09-24 |
| 分类 | Quantization（tier: major） |
| Label | documentation, ready, kv-connector |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 3 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/58472 |

## 一句话结论

DecodeBenchConnector 加 startup_fill 一次性填充 KV cache，M3 TP4 / GB200 同吞吐下 TTFT −20%、P99 ITL −11%。

## 动机

PR 针对 `DecodeBenchConnector` 做两处改动（各一个 commit）。其一修复 fp8 填充值：fp8 KV cache 以 `uint8` 存储，而原填充用 `torch.full(fill_mean, dtype=uint8)` 会截断为 0，导致 dummy prompt KV 全零，且随机填充（`fill_std > 0`）会抛异常。其二新增可选 `startup_fill`：当前 per-request 填充在接纳新请求的 decode step 内进行，每层都要把 block ID 拷到设备、用布尔 mask 过滤（一次 device sync）并构造临时张量，开销落在 decode 关键路径上。

## 优化点

fp8 修复：普通 fp8 `uint8` cache 现在按对应 fp8 dtype 编码填充值（平台 e4m3，`fp8_e5m2` 用 e5m2）；随机值在 float32 中采样并 clamp 到有限范围；带 scale/metadata 的 packed `uint8` 布局（如 `fp8_ds_mla`、NVFP4）保持零填充，并给出一次性 warning。`startup_fill`（`kv_connector_extra_config` 新增选项）：worker 在 `register_kv_caches` 时一次性填满整个 KV cache 并跳过 per-request 填充；circular-buffer 组、以及引擎会在用前把新块清零的 attention 组（hybrid Mamba、mixed precision）仍保留 per-request 填充，避免 dummy KV 被留成零；sleep mode 后不重填，默认行为不变。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py: 新增 _get_fp8_dtype（识别 uint8 存储的 fp8 cache）、_get_startup_fill_group_ids、startup_fill 逻辑与 docstring
tests/v1/kv_connector/unit/test_decode_bench_connector.py: 新增 fp8 读回、packed 布局保持零、startup fill（strided/padded/fp8）、无 per-step fill、circular-buffer 保留 per-request fill 等 14 个测试
docs/serving/expert_parallel_deployment.md: Benchmarking 段补充 startup_fill 用法说明
```

```python
- startup_fill (bool): Fill the whole KV cache once at startup instead
  of filling each new request's blocks in the step that schedules it
  (default: False). Circular-buffer caches are still zeroed per
  request, and attention caches whose new blocks the engine zeroes
  before use (hybrid Mamba or mixed-precision KV caches) are still
  filled per request. The fill is not redone after sleep mode
  discards the KV cache.
```

## 性能声明

测试条件（Decode-only A/B）：`--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"[,"kv_connector_extra_config":{"startup_fill":true}]}'`，`vllm bench serve --dataset-name random --ignore-eos`，warmup 后每模式 3 次取均值（min–max）。Qwen3-0.6B：1× GB200，3584 in / 256 out，concurrency 4；M3 TP4 GB200：`--language-model-only --block-size 128 --max-num-seqs 88 --max-num-batched-tokens 512 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`，32000 in / 256 out，concurrency 32。

| Model | Mode | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Median ITL (ms) | P99 ITL (ms) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | default | 1636 (1550–1682) | 70.8 (68.8–72.5) | 2.18 (2.11–2.31) | 2.15 (2.05–2.28) | 3.03 (2.86–3.19) |
| Qwen3-0.6B | `startup_fill` | 1839 (1782–1926) | 50.0 (47.0–51.6) | 1.99 (1.90–2.05) | 1.96 (1.87–2.02) | 2.59 (2.54–2.64) |
| M3 TP4 | default | 1273 (1265–1279) | 1477 (1382–1535) | 19.39 (19.13–19.65) | 11.55 (11.53–11.57) | 218.3 (216.3–219.5) |
| M3 TP4 | `startup_fill` | 1282 (1261–1303) | 1185 (1144–1242) | 20.34 (20.09–20.82) | 11.97 (11.77–12.25) | 193.3 (191.9–194.6) |

正文结论："M3: TTFT −20% and P99 ITL −11% at the same throughput. TPOT +4.9% because more requests are in the decode phase (24.6 → 26.0)"（perf claims 段标注 20% 与 P99 ITL −11% at the same throughput）。

## 验证与回归风险

验证：`pytest tests/v1/kv_connector/unit/test_decode_bench_connector.py` → 29 passed (14 new)，在 `main` 上 fp8 用例因全零或 `NotImplementedError` 失败；worker 侧 KV read-back（默认模式与 `startup_fill`）用 `Qwen/Qwen3-0.6B --kv-cache-dtype fp8`、`MiniMaxAI/MiniMax-M3-MXFP8` TP4（fp8 主 KV + fp8 indexer KV），读回值 0.015625 无非有限值，默认模式 5.3%(Qwen)/4.4%(M3) 元素被填充，`startup_fill` 99.5%(Qwen)/99.99%(M3)；Smoke runs：`Qwen3.8-Flash-Next-FP8`（circular buffer, TP4）200/200、`Falcon-H1-0.5B`（Mamba）100/100。可能影响路径：circular-buffer 与引擎清零的 attention 组仍走 per-request fill，sleep mode 后不重填，默认行为不变；PR 未注明更多回归风险。

diff 的 `decode_bench_connector.py` docstring 另注明 "The dummy KV cache is only meant for performance measurement; outputs are not meaningful for accuracy evaluation"，并说明 `startup_fill` 在 sleep mode 丢弃 KV cache 后不会重做填充；正文也写明 "The default behavior is unchanged"、KV read-back 的读回值为 0.015625 且无非有限值，以及 "No open PR covers either change"。正文还列出 Decode-only A/B 的统一命令形式 `--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"[,"kv_connector_extra_config":{"startup_fill":true}]}'` 与 `vllm bench serve --dataset-name random --ignore-eos`（warmup 后每模式 3 次）。

## 关联

#57751（正文提到其修复 default mode 下的 Mamba state filling，与本 PR 互补）；正文另称 "No open PR covers either change"，未给出其他前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
