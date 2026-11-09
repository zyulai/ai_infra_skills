# PR #44288: [Perf] detokenizer: Group return IPC sends by tokenizer worker

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hope1262946533 |
| 合并 | 2026-10-14 |
| 分类 | Parallelism / Communication (tier: major) |
| Label | documentation, run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/44288 |

## 一句话结论

按目的 worker 分组 Detokenizer→TokenizerWorker 回程 IPC 发送,发送次数减约 94%,整机输出吞吐 +2.1%。

## 动机

多 HTTP worker 模式下,DetokenizerManager 处理完一个 batch 输出后,原来按 request 逐条把结果发回各自所属的 TokenizerWorker(`http_worker_ipc`)。同一输出批内多个 request 常属于同一 worker,产生大量冗余 IPC send,return 侧 CPU 开销高。Router 侧的分组发送已由 #42238 合入 main;本 PR 修复 #43096,把同样的分组思路推进到最后一段回程。

## 优化点

新增默认关闭的开关 `SGLANG_OPT_DETOK_TOKENIZER_GROUP_SEND`(`EnvBool(False)`),在 multi-HTTP-worker detokenizer event loop 启动时读取一次。把 #42238 的分组逻辑泛化为 `_split_batch_by_key(output, ipcs, key_fn)`,按 `http_worker_ipc` 分组:同一目的地的多个 request 用 `_handle_output_by_indices` 拼成一个子 batch,每个目的地只发送一次。发送前先构造全部目的地 payload:splitter 抛 `_CannotSplitBatch` 时整批回退原 per-request 路径,不会发出一半;send 失败留在 fallback 之外,避免 fallback/重试重复发送已成功的 payload。每个 request owner 都不同时保持原路径,发送次数与 payload 不变。

## 关键改动

```text
python/sglang/srt/managers/multi_tokenizer_mixin.py: 新增 _split_batch_by_key/_send_output_to_tokenizer_workers,event loop 改走分组发送
python/sglang/srt/environ.py: 注册默认关闭的 SGLANG_OPT_DETOK_TOKENIZER_GROUP_SEND 开关
test/registered/unit/managers/test_multi_tokenizer_mixin.py: 分组、回退、顺序与默认关闭的回归测试
docs/docs/references/environment_variables.mdx: 开关文档(diff 节未展示,正文列出)
```

```python
def _send_output_to_tokenizer_workers(output, ipcs, socket_mapping) -> None:
    grouped = _split_batch_by_key(output, ipcs, lambda ipc_name: ipc_name)
    if grouped is None:
        for index, ipc_name in enumerate(ipcs):
            single_output = _handle_output_by_index(output, index)
            socket_mapping.send_output(ipc_name, single_output, is_tokenizer=True)
        return
    for ipc_name, grouped_output in grouped:
        socket_mapping.send_output(ipc_name, grouped_output, is_tokenizer=True)
```

## 性能声明

PR 正文基准条件:SGLang 0.5.17 @ `2948168`,DeepSeek-V4-Pro,TP8/DP8,无 MTP,32 tokenizer + 32 detokenizer workers,aiakperf streaming 随机数据集,并发 4960、79,360 请求、输入/输出 60,000/1,000、16 客户端进程。

- 无追踪整机客户端输出吞吐:三组三跑均值 +2.13%(禁用 42,297.97/42,564.88/42,306.00 token/s;启用 43,740.88/43,106.29/43,025.66 token/s);正文概称为 +2.1%。
- 开追踪 6 轮 A/B:+1.89%。
- 回程 IPC send 调用:79,376,257 次 → 4,790,103/4,325,743/5,063,191 次,减少约 93.62%-94.55%。
- detokenizer p50 CPU:288.56%/276.91%/289.95% → 107.51%/99.74%/105.75%;TokenizerWorker p50:731.38%/732.34%/730.86% → 373.87%/384.82%/364.64%(单核进程组 CPU 单位)。

PR 同时注明:收益依赖默认共享 listener 拓扑,per-worker listener(SO_REUSEPORT)下无收益(42,733 vs 42,764 token/s);不声称延迟均匀改善,median TTFT 33.12-39.85 s → 52.06-52.53 s,median E2EL 86.02-92.89 s → 104.96-105.31 s。

## 验证与回归风险

新增单元测试覆盖:非连续 owner 分组保持首目的地顺序与 per-目的地顺序及原所有权、唯一 owner 等价于原 per-request 路径、不修改输入对象、跨 message 不累积、pickle/msgpack 可选 wire 字段、token-ID 分组与 embedding 回退(shared splitter 仅覆盖 BatchTokenIDOutput/BatchStrOutput)、`_CannotSplitBatch` 整批回退、send 失败不重发已成功 payload、默认关闭与 event loop 开关行为;另跑 `test_msgpack_ipc_roundtrip.py`。风险:优化仅默认关闭时无影响,开启后收益强依赖 HTTP listener 拓扑;TTFT/E2EL 中位数劣化,需按 workload 评估。作者注明本地 CPU 环境无法导入 main,套件在 CI 复跑。

## 关联

前序依赖 #42238([Perf] Reduce multi-detokenizer router IPC sends,2026-10-09 合入;本 PR 复用其 helper,不改 Router 路径);修复 #43096。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
