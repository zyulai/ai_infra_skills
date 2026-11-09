# PR #60941: [Bugfix][HiSparse] Fix TP group hang on steps without scheduled tokens

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60941](https://github.com/vllm-project/vllm/pull/60941) |
| 作者 | vMaroon |
| 合并日期 | 2026-10-12 |
| labels | bug, hisparse, kv-connector, ready |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
HiSparse 共享 host pool 下,TP rank 的 host 落后一步且下一步无 scheduled token 时,共享事件记录顺序错乱导致 TP 组永久挂起;无 token 的 step 在 event wait 后加 TP CPU barrier。

## 问题现象
TP 组在没有任何 scheduled token 的 step 上永久挂起,且没有超时:典型表现为一个 TP rank 的 GPU 利用率 0%(stream 阻塞在 event wait),其余 7 个 TP peer 100%(在 all-reduce 里空转);开 DP+EP 时其他 DP rank 随后在 MoE collective 里阻塞(生产部署还观察到 DeepEP dispatch 超时)。生产场景为 HiSparse+NIXL 经 MultiConnector、TP8 x DP4、EP32 + MTP 的 P/D decode 部署,均发生在引擎启动头几分钟(NIXL 首次连接新 prefill 引擎使主线程 read post 慢至 0.1-2.4s)。双卡复现器:`zero_next`/`zero_now` 场景 main 上 3/3 HANG,本 PR 3/3 PASS。

## 根因
正文披露机制:协议用两个共享事件 A/B 交替,每个 step 各 rank 的 stream wait 上一步 rank 0 记录的事件,rank 0 在 `finish_forward` 记录另一个事件;stream wait 绑定的是 wait 调用时刻最近一次 record。触发条件:某 TP rank 的 host 落后 rank 0 一步,且下一步无 scheduled token。此时 rank 0 在无 forward 的 step 里立即记录事件,落后 rank 的 wait 绑定到了 rank 0 下一步的 record,而该 record 又排在 rank 0 自己 forward 5 的 all-reduce 之后(该 all-reduce 正等落后 rank)——循环等待。有 token 的 step 是安全的,因为 rank 0 在 record 前会在 staging 上 host-sync(`_finish_mirror_phase`),该 sync 排在前一个 forward 的 TP collective 之后;无 token 的 step 没有 forward,没有任何东西把 rank 0 的 record 排到其他 rank 的 wait 之后。

## 修复方案
`vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py` 的 `start_step`:在 `current_stream().wait_event(...)` 之后,若 `shared_host_region is not None and not num_tokens`,调用 `get_tp_group().barrier()`(CPU 组)。这样每个无 token step 上所有 rank 都先发出 wait 再让 rank 0 记录下一个事件。`num_tokens` 来自同一份 scheduler output,所有 rank 在相同 step 进 barrier;非共享 host region 时各 rank 用本地事件,不加 barrier。代价:每个无 token step 一次 CPU barrier(该类 step 本就没有 forward)。正文解释了为何不用 rank 0 单独 host-sync:`zero_now` 场景(无 token step 后跟有 token step)仍会挂。

```diff
         current_stream().wait_event(previous_host_write_event)
+        if self.shared_host_region is not None and not num_tokens:
+            # Without a forward, no TP collective keeps rank 0 from recording
+            # a shared event again before every rank has issued its wait.
+            get_tp_group().barrier()
```

## 影响与回归风险
- 影响窗口:#56629 引入跨 TP rank 共享 host pool 之后(正文提及,未给具体版本)。
- 回归面:仅 HiSparse connector 的 `start_step` 路径。共享 host region 的部署在每个无 token step 多一次 CPU barrier;有 token 的 step 完全不变;不共享 host region 的部署不加 barrier。新增单测覆盖 shared/num_tokens 四种组合。正文声明该补丁尚未在生产部署上端到端验证。

## 关联
- [PR #56629](https://github.com/vllm-project/vllm/pull/56629)(共享 host pool 引入)
- [PR #59877](https://github.com/vllm-project/vllm/pull/59877)(同一调查中发现的 NIXL 线程安全问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
