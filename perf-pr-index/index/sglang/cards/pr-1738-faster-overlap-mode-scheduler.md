# PR #1738: Faster overlap mode scheduler

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @merrymercy |
| 合并 | 2024-10-21 |
| 分类 | Overlap / Scheduler |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 1 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/1738 |

## 一句话结论

重排 overlap scheduler 的 kernel launch 与结果取回顺序,disable-radix 下吞吐 46.06 → 51.03 req/s(约 +10.8%)。

## 动机

overlap scheduler 的原实现里,forward 线程做完一次 forward 后要同步执行 `next_token_ids.tolist()`(阻塞式 D2H 拷贝)才把结果放入 output_queue;调度线程取结果的动作与下一 batch 的 kernel launch 互相串行阻塞,launch 与拷贝无法重叠。本 PR 改进 kernel launch 与 result fetching 的顺序,使 overlap 模式即使关闭 radix cache 也能带来 10% 吞吐提升;开启 radix cache 时预期收益更大。

## 优化点

拆出独立的 copy 线程,把"D2H 结果拷贝"从 forward 关键路径剥离:forward 线程在 `forward_batch_generation` 返回后立即对 `next_token_ids` 发起 `to("cpu", non_blocking=True)` 异步拷贝,record 一个 blocking CUDA event 放入 `copy_queue` 即继续处理下一 batch;copy 线程轮询 event 完成后再 `tolist()` 并写入 `output_queue`。这样上一 batch 的同步拷贝不再阻塞下一 batch 的 launch。同时 forward 线程维护 `has_inflight_batch` 与 `launch_event`(forward 发起完成后 set),`resulve_batch_result` 在有 in-flight batch 时先等 `launch_event`,保证调度线程取回上一批结果之前当前 batch 已被 launch,维持 CPU/GPU 流水。实现仍是 GIL 约束下的多线程(作者注明:多进程或关闭 GIL 可期望更大提升)。

## 关键改动

```text
python/sglang/srt/managers/tp_worker_overlap_thread.py: 新增 copy_thread/copy_queue 异步 D2H,launch_event 同步
```

```python
next_token_ids = next_token_ids.to("cpu", non_blocking=True)
copy_event = torch.cuda.Event(blocking=True)
copy_event.record()
self.copy_queue.put((copy_event, next_token_ids))
```

## 性能声明

meta-llama/Llama-3.1-8B-Instruct,bench_serving 3000 prompts(`--disable-radix`,硬件未注明):

- Overlap mode:51.03 req/s(input 11459.26 tok/s,output 9893.56 tok/s,Mean E2E 28986.97 ms,Mean TTFT 14495.13 ms)
- Normal mode:46.06 req/s(input 10342.28 tok/s,output 8929.19 tok/s,Mean E2E 31574.46 ms,Mean TTFT 15352.12 ms)
- 即 overlap 相对 normal 约 +10.8% 吞吐;作者称 radix cache 开启时预期更多加速

## 验证与回归风险

正文注明在 GSM-8k 上验证精度与 normal scheduler 一致;PR 未新增自动化测试。功能限制(作者注明):overlap scheduler 为实验特性,支持标准解码,但不支持 sampling penalizers(frequency/repetition penalty)与 constrained decoding(regex/JSON)。风险:`has_inflight_batch`/`launch_event` 为跨线程可变状态,事件时序错配可能让调度线程在 batch 未 launch 时提前消费结果;copy 线程以 1e-5 sleep 轮询 event,长队列下有忙等开销。

## 关联

前序:overlap scheduler 机制的最初引入(编号素材未注明);后续同类优化(如后续版本对 overlap 调度器的重构)素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
