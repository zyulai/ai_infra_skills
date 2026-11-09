# PR #34338: [perf] Collapse the DP attention scheduler sync to a single D2H copy

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-08-11 |
| 分类 | Overlap / Scheduler (major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 2 个文件(dp_attn.py 聚合消费与 two_batch_overlap.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/34338 |

## 一句话结论

把 DP attention 调度器的元数据同步从 5 次设备同步合并为一次整体 D2H 拷贝,每调度步省约 282 us。

## 动机

DP attention 下 scheduler 每步要消费 all-gather 回来的 info 张量(几百字节):原先对两列做 `.cpu()`、对三列做 `.min()`/`.max()` + `.item()` 归约、下游再 `.tolist()`,每次 reduction 都启动一个 kernel 并让调度线程阻塞在一个完整的 sync 往返上。payload 很小,成本全在同步延迟而非带宽。NCCL metadata-gather 路径受影响;默认 Gloo 路径张量本就在 host 上,此改动对其是 no-op。

## 优化点

把 gathered info 张量整体一次 `.cpu()` 拷到 host,再在 host 侧切片与归约:`global_num_tokens`、`global_num_tokens_for_logprob`、`can_run_decode_cuda_graph`、`is_extend_in_batch`、`can_run_prefill_cuda_graph`、`dp_cooperation_info` 与 TBO 的 `compute_output` 全部改为消费 host 张量。关键细节:不拷贝 `[:, 0, :]` 切片——当 `attn_tp * attn_cp > 1` 时该切片非连续,先切片会往阻塞等待里加一个 gather kernel;整体拷贝连续张量、host 侧切片可避免这约 30 us。`MLPSyncBatchInfo` 携带 `tp0_info_cpu` 替代设备张量,所有消费方都在 host 侧;`compute_output` 签名改为接收 `partial_global_info_cpu`,把 host 张量契约写进签名。继承自 #23011(superseded,原文件迁移后失效),但未 carry over int64→int32(实测 -1.0 us,噪声级)与预分配 buffer 单例(与现行进程状态约定冲突)。

## 关键改动

```text
python/sglang/srt/managers/scheduler_components/dp_attn.py: all_gather 消费改为单次整体 D2H + host 侧切片归约,MLPSyncBatchInfo 换 tp0_info_cpu
python/sglang/srt/batch_overlap/two_batch_overlap.py: compute_output 接收 host 张量,去掉自己的 .cpu()
```

核心改动:

```python
# One D2H for every field: each `.item()` / `.tolist()` on a device
# tensor is its own stream sync. Copy the whole tensor, not the
# `[:, 0, :]` slice -- that slice is non-contiguous once
# attn_tp * attn_cp > 1, adding a gather kernel inside the wait.
tp0_info_cpu = global_info_tensor.cpu()[:, 0, :]
self.tp0_info_cpu = tp0_info_cpu
self.global_num_tokens = tp0_info_cpu[:, 0].tolist()
self.can_run_decode_cuda_graph = bool(tp0_info_cpu[:, 2].min())
self.is_extend_in_batch = bool(tp0_info_cpu[:, 3].max())
```

## 性能声明

- 微基准(4 DP ranks,交错 A/B):消费 gathered 张量 454.6→172.0 us,即**每调度步省 282 us**。
- 消费成本按 attn_tp(单进程,绝对值偏低):attn_tp=1 时 212→49 us;attn_tp=2 时 289→62 us;attn_tp=8 时 256→52 us。先切片后拷贝会在非连续情形多留约 30 us 的 gather kernel。
- E2E(DeepSeek-V4-Flash,DP4 + EP + MTP,A/B/A 跨重启,4 并发):TPOT 7.90→7.63 ms(**−2.9%**),output throughput **+3.1%**;重复的 base run 与首次差异在 0.5% 内。
- 64 并发 E2E:无可分辨影响——机器漂移(+5.4%)大于信号;符合预期,因为节省的是每步固定开销,只在调度器 CPU 为关键路径时显现。

## 验证与回归风险

pack 中未见新增测试文件;验证手段为微基准与 A/B/A 重启式 E2E 对比。CI 状态:PR Test (Base) no_entry_sign、PR Test (Extra) 失败。风险面:仅 NCCL metadata-gather 路径生效,Gloo 路径 no-op;所有原 `tp0_info`(设备张量)消费点被改为 `tp0_info_cpu`,若有遗漏的设备侧消费会出错(签名契约已显式化);host 侧 `.min()`/`.max()` 语义与原 `.item()` 等价(bool 转换语义不变)。

## 关联

同区域:#23011(本 PR 的 supersede 前身,co-authored with 原作者);DP attention 调度同步、TBO(two-batch-overlap)准备路径相关 PR(pack 未列编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
