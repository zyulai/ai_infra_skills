# PR #42901: [Perf] deepseek-v4.1: Skip wasted layer 20 work on earlier prompt tokens under decoder bounded replay

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @vincentzed |
| 合并 | 2026-10-17 |
| 分类 | Speculative Decoding(tier: major) |
| Label | deepseek, run-ci |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/42901 |

## 一句话结论

DeepSeek-V4.1 decoder bounded replay 下让 layer 20 只对最后 128 tokens 跑主体计算,长 prompt TTFT 提速 5–10%。

## 动机

DeepSeek-V4.1 的 decoder bounded replay(`--enable-decoder-swa-bounded-replay`)让模型最后若干层在 prefill 时只处理每请求最后 128 个 prompt tokens,当前从 layer 21 开始,而 layer 20 仍处理全部 prompt token。Layer 20 的输出中只有最后 128 tokens 被 layer 21 读取,其在更早 tokens 上的 indexer、attention 与 MoE 工作全部被丢弃;layer 20 的 indexer 还要扫描全上下文,浪费随 prompt 长度增长。它唯一必须全量做的事是写 decode 之后要读的 compressed KV cache。

## 优化点

bounded replay 开启时,layer 20 先为每个 token 写 compressed KV cache(`write_tail_source_kv`,只跑 compressor、跳过 indexer),然后层主体只对最后 128 tokens 执行;写出的 compressed KV cache 与之前完全相同。Layer 20 的 indexer 结果改为直接为 tail 的 128 tokens 生成,原先把 tail 行从全尺寸缓冲拷入 tail metadata 的代码被整体删除。tail 起点从 `max(kv_source_layer_ids) + 1` 前移到该层本身(非 HIP,`tail_writes_source_kv`),并断言该层无 Engram、且同时是 index 与 candidate source;ROCm 保持旧行为(gfx950 fused boundary 一次调用跑整层,其 backend 在 tail 内携带 source 层的 top-k/candidates)。不加新 flag,bounded replay 关闭时零变化。兼容 #43736 的 prefill CUDA graph:graph 内准备 layer 20 输入的步骤(`_bcg_low_ratio_source_projections`)新增 `run_compressor`/`run_indexer` 参数,只计算调用者要求的部分。一个需 review 的行为变化:最后 128 tokens 上 layer 20 现在使用与 21–39 层相同的缩短滑动窗口(之前是全窗口),与 DeepSeek-V4.1 报告对 decoder replay 的描述("feed their encoder outputs through the decoder layers under the same SWA truncation")一致。

## 关键改动

```text
python/sglang/srt/layers/attention/deepseek_v4_backend.py: write_tail_source_kv、tail_source_kv_layer_id、删除 tail 行拷贝逻辑
python/sglang/srt/models/deepseek_v4.py: tail 起点前移到 last kv_source 层(非 HIP)并接入 write_tail_source_kv
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: ROCm 侧新增字段,保持旧 tail 行为
```

```python
def write_tail_source_kv(
    self, *, layer, x, positions, forward_batch: ForwardBatch
) -> None:
    """Write every row's compressed KV for the source layer the tail starts at,
    before enter_late_layer_tail; the layer's own pass inside the tail then
    skips its compressor."""
    self.forward_low_ratio_sources(
        layer=layer,
        x=x,
        q_lora=None,
        positions=positions,
        forward_batch=forward_batch,
        run_indexer=False,
    )
    self.tail_source_kv_layer_id = layer.layer_id
```

## 性能声明

4x B300,`deepseek-ai/DeepSeek-V4.1-Flash`,TP4+EP4,`--enable-decoder-swa-bounded-replay`。TTFT(单请求,3–5 次取中位,两列来自两次独立 server 启动):新 prompt 16K 139.6/139.5→132.8/131.6 ms(5%)、64K 577.5/564.9→526.6/524.5 ms(8%)、200K 2009.6/2015.4→1800.5/1811.9 ms(10%);500–8K 与各 cached 前缀续写行无变化("128 new tokens after 200K cached" 慢约 5 ms,作者判断为启动间噪声)。layer 20 耗时(16K chunk,per-layer GPU sync):prompt 早期 11.9→3.3 ms;200K prompt 后期 27.0→5.2 ms。合并 main `684fea8af3` 后重测:14,795 tokens no graph 6%、prefill graph 8%;6,318 tokens graph 7%。合并 `abf92826ee` 后(含 TRT-LLM):14,795 FlashMLA 6%、TRT-LLM 8%;6,318 FlashMLA 8%、TRT-LLM 4%。GSM8K 1319 题:各配置 0.897–0.914,与 main 差异在 ±0.007 的运行噪声内。

## 验证与回归风险

多轮 GSM8K 1319 题(无/有 prefill graph、FlashMLA/TRT-LLM、DSpark block size 5);`test/registered/unit/layers/test_dsv4*.py` 70/71 passed;各轮无 scheduler 异常。未测试:prefill CP + bounded replay;ROCm(行为保持不变)。回归面限于 bounded replay 开启且非 HIP 的路径;行为变化为 tail 层(含 layer 20)使用缩短 SWA 窗口。作者标注 follow-up:`CandidateMetadata.tail` 现仅被自身测试调用,可由后续 PR 删除。pack 中 CI:Base 与 AMD ROCm 10 进行中,Extra 失败标记。

## 关联

前序:#43736(decoder bounded replay 的 breakable prefill CUDA graph)、#44326(TRT-LLM 后端 + bounded replay under graph)、#43607(DSV4 layer 代码拆分为 stage 文件)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
