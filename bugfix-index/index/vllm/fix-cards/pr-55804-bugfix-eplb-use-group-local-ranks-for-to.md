# PR #55804: [Bugfix][EPLB] Use group-local ranks for torch P2P transfers

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55804](https://github.com/vllm-project/vllm/pull/55804) |
| 作者 | LostFox11 |
| 合并日期 | 2026-10-06 |
| labels | bug, ready, verified |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
EPLB 在流水线并行下把 EP 组内局部 rank 当全局 rank 传给 `P2POp.peer`,权重传输报错;改用 `group`+`group_peer` 参数修复 send/recv 两处。

## 问题现象
使用 pipeline parallelism 时,EPLB 专家权重传输失败。torch NCCL 与 Gloo 两种后端均报:

```text
ValueError: Global rank {global_rank} is not part of group {group}
```

触发条件:EPLB 向其 communicator 传入的是 EP 组内局部 peer rank,而 torch 后端将其作为 `P2POp.peer`(语义为全局 rank)。例如 EP 组包含全局 rank `[2, 3]` 时,局部 peer `1` 被解释成全局 rank `1`,该 rank 不在组内,expert-weight transfer 失败。

## 根因
PR 正文披露:EPLB 传给 communicator 的是 EP-group-local peer ranks,但 torch NCCL 和 Gloo 后端把它作为 `P2POp.peer` 使用,而 `P2POp.peer` 期望全局 rank;组内局部 rank 与全局 rank 语义混淆导致点到点传输指向组外 rank 而失败。

## 修复方案
在 `vllm/distributed/eplb/eplb_communicator.py` 中,对 send 与 receive 两处 `torch.distributed` communicator(含 `add_send`/`add_recv` 的 `self._ep_group` 以及 `build_ops` 中 CPU 组 `self._cpu_group` 共四个调用点)改用显式 `group=` 与 `group_peer=` 参数:

```diff
                 P2POp(
                     torch.distributed.isend,
                     tensor,
-                    dst_rank,
-                    self._ep_group,
+                    group=self._ep_group,
+                    group_peer=dst_rank,
                 )
```

即把位置参数 `(peer, group)` 改为关键字参数 `group=...、group_peer=...`,使局部 rank 通过 `group_peer` 语义传给 torch。另在 `tests/distributed/test_eplb_execute.py` 新增 `test_eplb_communicator_pipeline_parallel` 回归测试,覆盖 PP=1/PP=2、每 EP 组两 rank、`torch_nccl`/`torch_gloo` 两种后端(四 rank 场景覆盖非零 rank 组 `[2, 3]`)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:改动限于 EPLB communicator 的 P2P op 构造(EP 组与 CPU 组两条路径);修复后传输语义由 `group_peer` 承担,若运行环境 torch 版本不支持 `P2POp` 的 `group_peer` 参数,该路径会受影响。作者验证:修复后两种后端测试通过,且以 Gloo 后端对 Qwen3.6-35B-A3B 做 GSM8K 评估达 96.51% 准确率。

## 关联
正文提到相关 EP topology PR #52099(未修改该 communicator 文件),以及 2026-09-08 的重复工作排查(未发现相同修复的 open PR);未给出链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
