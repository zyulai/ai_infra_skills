# PR #40703: [PD] Add opt-in prefill-complete decode KV allocation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ekrim |
| 合并 | 2026-10-03 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | documentation, run-ci, run-ci-extra, release-highlight, bypass-fail-fast, highest-priority |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 可用,另有 11 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/40703 |

## 一句话结论

PD 分离新增 opt-in 的 prefill_complete KV 分配策略,decode KV 预留推迟到 prefill 完成;混合负载下系统输出吞吐 +27%、TTFT p95 −49%。

## 动机

PD 分离服务中,decode 会在该请求的 prefill 仍在运行时就预留其目标 KV slots;长 prefill 因此占据其他请求本可使用的 decode 容量。目标是 decode KV 容量紧张且 prefill 完成时间差异大的 workload(例如长短 prompt 并发):unfinished 长 prefill 提前占住 decode slot,推迟了已就绪请求。这是分配时机的改变——延迟预留使容量在 prefill 结束前保持可用,并不让分配本身或 decode kernel 变快。

## 优化点

- 新增 `--disaggregation-decode-allocation-policy prefill_complete`(默认 `early`,opt-in,须在两个角色上同时设置):per-request 目标 KV 预留推迟到 accepted final prefill result 就绪;全局 KV pool 的创建与大小不变。
- 新增 Mooncake readiness control 协议:`DeferredBootstrap` 房间表带 request ownership/nonce、订阅重试 backoff、bounded waits、失败传播与 pre-allocation 取消,每个 room 标识一个请求,迟到端点面对有界 tombstone;表本身不做任何 I/O。
- prefill 在接受最终 prefill result 并备好 output/logprob/speculative sidecars 后,经 `CommonKVSender.mark_prefill_complete` 发 readiness 信号;decode 把 admission gate 挂在该信号上。readiness 用普通状态信道传输(`KVPoll.WaitingForInput`),不授权写入。
- 保留既有 destination-metadata 握手与 abort/drain 协议;optimistic retry 耗尽仍允许 prefill;true retraction/rebootstrap 的重计算在等待 source readiness 之前派发;`_prefill_wait_start` 把 decode 的等待预算从 prefill 排队/计算时间中剥离。
- 配置校验(`_validate_prefill_complete`):仅支持 Mooncake(含 TCP)、attention TP1/CP1/PP1(允许 attention DP)、prefill HiCache 仅 L2 write_back;排除 decode radix/HiCache、staging、runtime role switching、PD multiplexing、encoder 分离、decoupled speculation、HiSparse 等;prefill 侧把 optimistic attempts 0 解析为 1,被配置禁用 optimistic prefill 的组合直接拒绝。

## 关键改动

```text
python/sglang/srt/arg_groups/fields/disagg.py: 新增 --disaggregation-decode-allocation-policy (early|prefill_complete)
python/sglang/srt/disaggregation/common/bootstrap.py: 新增 DeferredBootstrap 就绪控制表(ownership/nonce/超时清理)
python/sglang/srt/disaggregation/common/conn.py: bootstrap 消息处理、readiness 通知、receiver poll/超时重构造
python/sglang/srt/arg_groups/pd_disaggregation_hook.py: _validate_prefill_complete 拒绝不支持的配置组合
python/sglang/srt/disaggregation/decode.py: _init_receiver 推迟 admission,recompute 派发时机按策略分流
```

## 性能声明

Historical GLM-5.2-NVFP4 ABBA 对比:real MTP3,12 块 Rubin GPU,2 个 prefill replica + 1 个 decode replica(各 TP4/DP4/EP4),FP8 KV,Mooncake;decode KV 上限刻意压到 163,840 tokens/rank 以制造容量压力(非生产默认配置)。每臂先 64 个短请求(ISL 4,096 / OSL 256 / 并发 8),再 256 个混合请求(64×ISL 131,072 + 192×ISL 4,096,OSL 256,并发 64);下表为混合负载、两次运行的算术平均变化:

- Output tokens/s(系统):458.88/435.98 → 616.39/518.39,+26.81%。
- TTFT p95:71.22/79.97 s → 36.03/41.27 s,−48.87%。
- Mean TPOT:4.108/3.869 → 4.204/3.943 ms,+2.13%。
- ITL p99:34.40/12.03 → 46.40/43.10 ms,+92.77%(尾部明显变差)。

吞吐/TTFT 收益伴随更差的 ITL 尾部;延迟分配也让 source KV 保留更久并消除 chunked prefill 与 KV 传输的 overlap。另一组 Qwen3-8B policy-only 对比吞吐差约 1%(run variation 之内),无明显短请求尾部改善——这是策略保持 opt-in 的原因。PR 注明:更大的 workload campaign 未通过最终 provenance audit,其数字未被采信。

## 验证与回归风险

测试:99 个 CPU tests + 27 subtests 通过(Linux ARM64,六个文件,Mori 依赖的一个跳过);两 GPU 生命周期套件 `test/registered/e2e/disaggregation/test_prefill_complete_allocation.py` 覆盖 multi-chunk prefill、prefix reuse、并发、传输前/后取消、retract/resume、optimistic retry 耗尽、timeout cleanup;rebase 前通过 8 个 registered GPU lifecycle 测试、attention-DP 附加检查与 6 个 TCP/prefill-HiCache 检查;GLM 四臂 functional parity(1,280 请求全完成,max logprob diff 0.002925,PR 注明非模型质量基准)。遗留:最新 rebase 后的 CPU/GPU 回归与性能验证标记为 Pending(本地 CPU 因缺 torchvision 未收集);要求先以 `early` 升级全部通信 worker 再切换策略(旧 decode worker 无法执行 policy-mismatch 检查)。风险面:策略不适配即启动失败;仅支持 full-attention 与 DSA KV pool;room 复用依赖 worker cleanup 顺序与 `_owns_bootstrap_room` 语义。

## 关联

无(pack 未给出同区域其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
