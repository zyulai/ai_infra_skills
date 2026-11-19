# PR #50895: [Perf] disaggregation: Stream a PD decode request's first token with its first decode output

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-11-17 |
| 分类 | Overlap / Scheduler(major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/50895 |

## 一句话结论

PD decode 服务器不再在请求准入时立刻流式发送 prefill 采样的首 token,而是让它随第一个 decode 输出一并发出:GB300 上 DeepSeek-V4.1 DEP32 decode 到第二个 token 的时间从约 104 ms 降到 92–95 ms,三个部署点 p90 TPOT 改善 3–6%,吞吐持平。

## 动机

PD decode 服务器在请求准入时即通过 `process_batch_result_prebuilt` 流出 prefill 采样的首 token;在 overlap scheduler 下,该请求的第一个 decode forward 排在已在飞的 forward 之后,后续 token 要晚约两个 decode step 才到。客户端 TPOT 按 (e2e − TTFT)/(osl − 1) 计算,这段空隙被记入每个请求的 TPOT:DeepSeek-V4.1 DEP32 decode 上,第二个 token 的时间约 104 ms,而稳态 32-token chunk 只要约 92 ms。自己在自己首个 decode step 采样首 token 的 decode 服务器没有这个空隙。

## 优化点

- 准入时只对"已完成"的请求立即流出(例如 `max_new_tokens=1`);运行中请求的 prefill token 与其第一个 decode 输出一起发出——output streamer 因 `send_token_offset == 0` 会立即发送。
- 仅影响 PD decode;非 disaggregated 服务路径不变。输出 token 内容不变,只改首 token 发出的时机。

## 关键改动

```python
# python/sglang/srt/managers/scheduler_components/batch_result_processor.py
@@ -170,8 +170,11 @@ def process_batch_result_prebuilt(self, batch: ScheduleBatch):
+        # Running requests send the prefill token with their first decode output.
         # Note: Logprobs should be handled on the prefill engine.
-        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
+        self.output_streamer.stream_output(
+            [req for req in batch.reqs if req.finished()], batch.return_logprob
+        )
```

diff 仅此一处:把无差别 stream_output 过滤为仅已结束请求。

## 性能声明

(全部照抄 PR 正文,未独立核实)

- 环境:GB300;DeepSeek-V4.1 带 MTP;PD disaggregation;agentic multi-turn 负载,每 run 3600 s。三个部署点:
  - 10 prefill + DEP32 decode,并发 4000:吞吐 620.4k→618.5k(−0.3%),p50/p90 TPOT 2.87/3.05→2.82/2.95 ms,TPOT@1/p90 327.9→339.0(+3.4%),p50/p90 TTFT 1.92/6.25→2.03/6.43 s。
  - 24 prefill + DEP32 decode,并发 10288:吞吐 1094.6k→1101.4k(+0.6%),p90 TPOT 5.24→4.95 ms,TPOT@1/p90 190.8→202.0(+5.9%)。
  - 14 prefill + DEP16 decode,并发 6600:吞吐 1208.6k→1206.0k(−0.2%),p90 TPOT 7.31→7.09 ms,TPOT@1/p90 136.8→141.0(+3.1%)。
- 结论原话:"Throughput is flat within run-to-run noise at every point, and p90 TPOT improves by 3-6%。"
- 机制确认(4 次 1200 s 并发 4000 run):到第二个 token 时间 p50 约 104 ms → 92–95 ms;TPOT p50 −1.5% 至 −3.7%;decode step time 不变(9.82–9.92 ms vs 9.95 ms);TTFT +约 20 ms(一个 decode step),在该负载的 TTFT 噪声内。

## 验证与回归风险

改动一行、语义简单:只有未完成请求的首 token 延后到第一个 decode 输出。PR 的 Accuracy Tests 声明输出 token 不变。Checklist 中单元测试与文档两项未勾选;CI 三个 run(Base/Extra/AMD)均显示失败图标(pack 未提供失败细节)。风险面:`max_new_tokens=1` 之外若存在其他"准入即完成"的语义边界(如 logprob 处理)依赖原先行为,需要回归;非 PD 路径不受影响。

## 关联

`process_batch_result_prebuilt`(prebuilt 批结果处理路径)、output streamer 的 `send_token_offset == 0` 立即发送语义。pack 未提供更多关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
