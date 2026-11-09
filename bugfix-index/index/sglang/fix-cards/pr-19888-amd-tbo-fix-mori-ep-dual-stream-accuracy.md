# PR #19888: [AMD][TBO] Fix mori ep dual stream accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19888](https://github.com/sgl-project/sglang/pull/19888) |
| 作者 | billishyahao |
| 合并日期 | 2026-03-18 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Overlap / Scheduler |
| bug 类型 | correctness |

## 一句话结论
TBO 双流下多个 MoriEPDispatcher 共用同一 mori op 致精度错,现按 instance_id 把 mori op 与对应流配对修复。

## 问题现象
开启 two-batch-overlap(TBO)且 MoE a2a 后端为 mori 时出现精度问题(mori ep dual stream accuracy)。正文未描述具体现象细节与触发日志;验证环境为 DSR1 PD 分离(preill/decode 双实例)TP8 DP8 EP8、`--moe-a2a-backend mori --enable-two-batch-overlap --enable-dp-attention`,GSM8K 200 题测得 Accuracy 0.975、Invalid 0.000。

## 根因
正文仅说明修复手段:"adding the instance id to pair the mori op to according stream",未展开根因分析。

从 diff 看:TBO 为双流构建多个内层 `MoriEPDispatcher`,原先每个 dispatcher 初始化时都以固定名字 `"mori"` 注册进程组并初始化 mori shmem,即多个实例共用同一个 mori op/进程组绑定,双流场景下操作与流不配对(由 diff 推断);新代码对 "already registered" 容忍并跳过重复的 shmem init(注释:新注册才需 init mori shmem,已注册则复用前一个),并为每个实例传 `instance_id` 区分配对。

## 修复方案
1. `batch_overlap/two_batch_overlap.py`:构建 mori 内层 dispatcher 时传入 `instance_id=i`:

```diff
         elif get_moe_a2a_backend().is_mori():
             self._inners = [
-                MoriEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)
+                MoriEPDispatcher(instance_id=i, **kwargs)
+                for i in range(num_inner_dispatchers)
             ]
```

2. `layers/moe/token_dispatcher/moriep.py`:`MoriEPDispatcher`/`init_mori_op` 增加 `instance_id` 参数;进程组注册改为容错("already registered" 忽略、其余 raise),仅在新注册时才执行 `mori.shmem.shmem_torch_process_group_init`。
3. dispatch_a/dispatch_b 重构:FP8/FP4 量化移入 `dispatch_a` 并随接口返回 `scale`/`output_dtype`,`dispatch_b` 只负责真正的 mori dispatch(接口相应扩展)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:mori a2a 后端的 dispatcher 初始化与 TBO 双流路径(instance_id 配对、shmem init 去重);dispatch_a/b 接口签名变化影响所有调用该 dispatcher 组合的路径;非 mori 后端(deepep/nixl)分支未动。GSM8K 0.975 为修复后的精度结果,正文未给修复前对照数字;正文附双流 trace 截图。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
