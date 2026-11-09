# PR #13681: [perf] Add two stream norm for `Olmo3` speedup 5%

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @vincentzed |
| 合并 | 2026-01-08 |
| 分类 | Model-specific Fast Path(major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供统计);diff 实际仅 1 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/13681 |

## 一句话结论

Olmo3/Olmo2 的 q/k RMSNorm 在 CUDA graph capture 时拆到双 stream 重叠,吞吐约 +7%。

## 动机

Olmo2 架构(olmo2.py 同时服务 Olmo3)的 `_apply_qk_norm` 在 attention 内对 q、k 依次做 `q_norm`、`k_norm`,两个独立的 RMSNorm kernel 串行执行。此前 #7740 与 #5977 已为其他模型引入 alt_stream 重叠模式,证明该手法有效;Olmo 的两个 norm 之间无数据依赖,是天然的重叠候选。作者也提到 fetch fast stream 的做法,但收益差异很小。

## 优化点

模型构造链(ForCausalLM → Model → DecoderLayer → Attention)全程透传 `alt_stream: Optional[torch.cuda.Stream]`;CUDA 上若无外部传入则在模型顶层新建一条 stream。`_apply_qk_norm` 中,当 `alt_stream` 存在且处于 CUDA graph capture 模式时:`alt_stream.wait_stream(current_stream)` 后,当前 stream 算 `q_norm`,alt_stream 算 `k_norm`,最后 `current_stream.wait_stream(alt_stream)` 汇合;非 capture 或无 alt_stream 时保持原来的 `forward_native` 串行路径。限定了 TP>1 时先 all-gather 再 norm 的原有顺序不变。仅在 capture 模式启用可保证 graph replay 时双 stream 拓扑被固化。

## 关键改动

```text
python/sglang/srt/models/olmo2.py: alt_stream 贯穿构造链 + _apply_qk_norm 双 stream 重叠 q/k norm
```

```python
if self.alt_stream is not None and get_is_capture_mode():
    current_stream = torch.cuda.current_stream()
    self.alt_stream.wait_stream(current_stream)
    q_by_last = self.q_norm(q.reshape(-1, q.shape[-1]))
    with torch.cuda.stream(self.alt_stream):
        k_by_last = self.k_norm(k.reshape(-1, k.shape[-1]))
    current_stream.wait_stream(self.alt_stream)
```

## 性能声明

单卡(CUDA_VISIBLE_DEVICES=6),allenai/Olmo-3-7B-Instruct,bench_serving random 4096 输入/1024 输出,512 prompts,rate inf,flush-cache:

| 指标 | Before | After |
| --- | ---: | ---: |
| Benchmark duration (s) | 82.42 | 77.05 |
| Input token throughput (tok/s) | 12720.70 | 13606.65 |
| Output token throughput (tok/s) | 3182.03 | 3403.64 |
| Total token throughput (tok/s) | 15902.73 | 17010.29 |
| Mean E2E Latency (ms) | 46918.92 | 44309.72 |
| P99 TTFT (ms) | 81733.94 | 76490.15 |

作者总结:主吞吐指标约 +7%,mean latency 约 -5.6%,约 1.07x 加速。PR 未注明 GPU 型号。

## 验证与回归风险

PR 正文的 Accuracy Tests 与 Benchmarking 章节为空,未提供准确性与数值等价验证;正确性依赖 stream 事件的 wait_stream 语义保证。回归风险:双 stream 重叠只在 CUDA graph capture 模式下生效,非 capture eager 路径无收益;若 norm kernel 资源饱和或 kernel 太短,重叠收益会在其他硬件上缩水;TP>1 时 all-gather 后的切分逻辑与 stream 汇合顺序耦合,改动仅限 olmo2.py,影响面为 Olmo2/Olmo3 系列。

## 关联

同手法前序:#7740、#5977(其他模型的 alt_stream 重叠)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
