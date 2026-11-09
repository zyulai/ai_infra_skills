# PR #18933: [gRPC] Fix scheduler startup broken by context parallel refactor

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18933](https://github.com/sgl-project/sglang/pull/18933) |
| 作者 | slin1237 |
| 合并日期 | 2026-02-17 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论
#17213 重构给 run_scheduler_process 新增必填参数,gRPC 启动器未同步致启动失败;补齐 rank 计算与传参。

## 问题现象
gRPC server 启动失败。触发条件:走 gRPC 入口启动,且代码已包含 #17213(context parallel 状态重构)。该重构把 `attn_cp_rank` 与 `moe_dp_rank` 设为 `run_scheduler_process` 的新必填参数,但 gRPC 的 `scheduler_launcher.py` 未同步更新,启动时以错误的参数调用调度进程。正文未附具体报错日志。

## 根因
正文披露:PR #17213(refactor context parallel state)在 `run_scheduler_process` 中新增必填参数 `attn_cp_rank`、`moe_dp_rank`,gRPC 的 `scheduler_launcher.py` 没有跟随更新,调用时参数不匹配,导致 gRPC server 启动失败。属重构后调用点未同步更新的回归。

## 修复方案
改 `python/sglang/srt/grpc/scheduler_launcher.py` 的 `launch_scheduler_process_only`:
1. 按 `engine.py` 的逻辑计算 `attn_dp_size`/`attn_tp_size`,进而得到 `attn_cp_rank` 与 `moe_dp_rank`;
2. `moe_ep_rank` 计算改为计入 `moe_dp_size`;
3. 以正确的位置顺序把 `attn_cp_rank`、`moe_dp_rank` 传入 `run_scheduler_process`。

```diff
+                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
+                moe_dp_rank = tp_rank // (
+                    server_args.tp_size // server_args.moe_dp_size
+                )
                 moe_ep_rank = (
                     tp_rank
                     % (server_args.tp_size // server_args.moe_dp_size)
@@ -123,6 +141,8 @@ def launch_scheduler_process_only(
                         tp_rank,
+                        attn_cp_rank,
+                        moe_dp_rank,
                         moe_ep_rank,
```

## 影响与回归风险
- 回归面:仅改 gRPC 启动路径(`grpc/scheduler_launcher.py`)的 rank 计算与进程传参;engine.py 等非 gRPC 路径不受影响。新增计算逻辑与 engine.py 对齐;若 `attn_cp_size`/`moe_dp_size` 等 server_args 在 gRPC 场景下取值异常,计算出的 rank 会随之偏差。CP / DP-Attention / MoE 并行组合下的 rank 划分正确性依赖这次对齐。

## 关联
- [#17213](https://github.com/sgl-project/sglang/pull/17213)(正文提及的 context parallel 状态重构 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
