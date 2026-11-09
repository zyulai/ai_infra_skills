# PR #22497: fix prefill tps log accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22497](https://github.com/sgl-project/sglang/pull/22497) |
| 作者 | ZhaiFeiyue |
| 合并日期 | 2026-04-12 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

prefill TPS 日志用上一批 token 数除以当前批时间间隔,产生最高 3 倍虚高尖峰与空闲后错误读数,改为当前批 token ÷ 当前间隔。

## 问题现象

prefill 的 `input throughput (token/s)` 日志指标出现虚假尖峰:最高达 3 倍虚高,空闲一段时间后的读数也不正确。作者在 1P1D PD disaggregation(DeepSeek-R1 MXFP4,EP4 prefill + EP8 decode,ISL=1024/OSL=1024,10240 prompts)实测:Max TPS 修前 90,527(纯虚假),修后 39,666;Mean 6,331→6,025,Median 3,657→3,754(P95 19,684→18,198,P99 31,998→30,625),稳态下两公式几乎一致,差异只在尖峰。

## 根因

正文按三个阶段梳理:原始实现 accumulator 从未累加,TPS 恒为 0.00;#7245(c4943867)用「存当前批 token、下次除」的绕过方式让其非零,却引入分子/分母错位——token 数来自 batch N-1,而 `gap_latency` 度量 batch N 的墙钟间隔。prefill 相邻批次的 `#new-token` 常不相同,TPS 便失真。作者举例:batch N-1 为 12288 token/1.0s,batch N 为 1024 token/0.08s 时,旧公式报 12288/0.08=153,600 token/s(虚高 15 倍),新公式 1024/0.08=12,800 token/s(正确);反过来 batch N token 多、N-1 token 少时则低估。

## 修复方案

改 `python/sglang/srt/observability/scheduler_metrics_mixin.py` 的 `report_prefill_stats`:用当前批的 `log_input_tokens` 除以当前 `gap_latency`(gap>0 才除,否则 0.0),与 decode 侧 `gen throughput` 的口径一致;`time.perf_counter()` 只取一次避免两次取值的偏差。同时删除已无用的 `self.last_prefill_tokens` 字段:

```diff
-        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
-        self.last_prefill_stats_tic = time.perf_counter()
-        self.last_input_throughput = self.last_prefill_tokens / gap_latency
-        self.last_prefill_tokens = prefill_stats.log_input_tokens
+        now = time.perf_counter()
+        gap_latency = now - self.last_prefill_stats_tic
+        self.last_prefill_stats_tic = now
+        self.last_input_throughput = (
+            prefill_stats.log_input_tokens / gap_latency if gap_latency > 0 else 0.0
+        )
```

## 影响与回归风险

- 回归面:仅影响 prefill 吞吐日志/指标的数值口径,不影响任何实际计算(作者确认);读该日志做监控或基准的下游会看到 max/尖峰显著下降、稳态值不变。

## 关联

- #7245(引入 token/time 错位的历史 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
