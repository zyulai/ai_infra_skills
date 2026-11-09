# PR #38779: Fix: post-load staging regression breaks offload meta/sharded_gpu modes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38779](https://github.com/sgl-project/sglang/pull/38779) |
| 作者 | Hide-on-bushsh |
| 合并日期 | 2026-09-21 |
| labels | quant, run-ci |
| 热度 | 👍 0 · 💬 18 |
| 分类 | KV Cache / Memory |
| bug 类型 | regression |

## 一句话结论
#35180 起 staging 对 offloader 停放在 meta 的参数直接报错,scheduler 启动即死;改为入口即 meta 的张量原样放行,处理中变 meta 的仍报错。

## 问题现象
offloader V2 的两种模式下启动即崩:崩溃日志(正文照抄)`RuntimeError: Cannot post-process meta tensor FusedMoE.w13_weight`,scheduler 在启动期间死亡。触发条件:①`--offload-mode=meta`:所有 offloaded 参数在 build 时变 meta;②`--offload-mode=sharded_gpu` / `shm_cpu`:非 rank-0 DP rank 在加载期间把参数停在 meta(rank 0 持有真实数据)。该 bug 与设备无关(CUDA/NPU 同一代码路径),由 NPU offload CI 套件(`test_npu_offload_modes.py`、`test_npu_offload_group_size.py`)捕获。

## 根因
正文披露:offloader 有意让 offloaded 参数以 meta 状态穿过 weight post-processing——device kernel 会跳过 meta 参数(如 `npu_format_cast` 有显式 meta skip),旧的 `device_loading_context` 对 meta 张量原样放行;#35180 起 `process_weights_after_loading` 的新的 upfront 检查对 meta 张量直接 raise,破坏了这一契约。

## 修复方案
`model_loader/post_load.py::stage_module_for_post_load()`:入口快照时把已是 meta 的张量记录进 `meta_tensor_ids` 并跳过(不 staging、不 restore),restore 阶段同样跳过,恢复 pre-#35180 行为;「处理过程中才变成 meta」的张量仍 raise,真实 kernel bug 继续 fail-fast。

```python
         if tensor.is_meta:
-            raise RuntimeError(
-                f"Cannot post-process meta tensor {type(owner).__name__}.{name}"
-            )
+            meta_tensor_ids.add(id(tensor))
+            continue
```

## 影响与回归风险
- 影响窗口:#35180 之后至本修复
- 回归面:仅 `stage_module_for_post_load` 对入口即 meta 张量的处理;对 post-load 期间新产生的 meta 张量的保护检查保留,原 fail-fast 语义未被削弱。单测相应改为「parks existing meta」与「rejects meta produced by post-load」两条。

## 关联
- #35180(引入 upfront 检查的变更)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
