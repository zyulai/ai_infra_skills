# PR #21170: [BUGFIX] Fix CP residual size mismatch crash when tp_size == attn_cp_size

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21170](https://github.com/sgl-project/sglang/pull/21170) |
| 作者 | yudian0504 |
| 合并日期 | 2026-03-23 |
| labels | high priority |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
tp_size == attn_cp_size 时 FULL 与 SCATTERED 组大小相同,通信被误判为 no-op,residual 尺寸不匹配(885 vs 7080)崩溃;调整路由判断顺序修复。

## 问题现象
合并 commit `bb737d7a829bac8fb6386fb6f494e1b2403c598f`([PR #18233](https://github.com/sgl-project/sglang/pull/18233))后,GLM-5/DS-V3.2 开启 context parallelism(CP)推理即崩溃,报:
```
RuntimeError: Check failed: residual.size(0) == batch_size (885 vs. 7080)
```
完整堆栈显示崩溃发生在 `ATTN_CP4 TP4` 配置的 scheduler 事件循环中,路径为 `deepseek_v2.py` 层 forward → `layer_communicator.prepare_attn` → `fused_add_rmsnorm`(flashinfer norm.cu 的 `TVM_FFI_ICHECK_EQ(residual.size(0), batch_size)` 断言)。正文指出:8 倍失配恰好对应 `--attn-cp-size 8`。

## 根因
正文完整披露:commit bb737d7a 将 `CommunicateContext.init_new()` 中 `ScatterMode.FULL` 的 process group 大小从 `tp_size` 改为 `tp_size // attn_cp_size`(CP 开启时「full」组不含 CP rank,概念上正确);但当 `tp_size == attn_cp_size`(如都为 8)时,`FULL` 与 `SCATTERED` 的组大小同为 1,`NSACPCommunicateSummableTensorPairFn.get_fn()` 的路由逻辑用 `is_same_group_size()` 判断是否需要通信,两者相等时错误地返回 `_trivial`(no-op),跳过了 MLP 之后 hidden_states 的 CP reduce-scatter。级联过程:
1. `prepare_mlp()` 将 hidden_states all-gather 到全量(7080),residual 保持 CP 本地(885);
2. `postprocess_layer()` 本应 reduce-scatter 回去,`_trivial` 原样透传;
3. 下一层 `prepare_attn()` 收到 7080 vs 885 的不匹配张量;
4. `fused_add_rmsnorm` 断言失败。

## 修复方案
`python/sglang/srt/layers/communicator_nsa_cp.py` 的 `NSACPCommunicateSummableTensorPairFn.get_fn()`:把「`hidden_states` 输入为 `FULL` 且 residual 输入为 `SCATTERED` 时返回 `_scatter_hidden_states`」的精确枚举匹配检查移到基于组大小的 `_trivial` 快捷判断之前:
```diff
-        if context.is_same_group_size(
-            hidden_states_input_mode, output_mode
-        ) and context.is_same_group_size(residual_input_mode, output_mode):
-            return NSACPCommunicateSummableTensorPairFn._trivial
-
         if (
             (hidden_states_input_mode == ScatterMode.FULL)
             and (residual_input_mode == ScatterMode.SCATTERED)
...
+        if context.is_same_group_size(
+            hidden_states_input_mode, output_mode
+        ) and context.is_same_group_size(residual_input_mode, output_mode):
+            return NSACPCommunicateSummableTensorPairFn._trivial
```
即使组大小巧合相等,`FULL` 与 `SCATTERED` 在 CP 下代表不同的数据布局,必须显式 scatter。

## 影响与回归风险
- 影响窗口:自 PR #18233 对应 commit 合并起,至本 PR 之前;触发条件为 `tp_size == attn_cp_size` 的 CP 部署。
- 回归面:仅改 `communicator_nsa_cp.py` 中 NSA CP 通信路由的判断顺序;`FULL→SCATTERED` 组合始终走显式 scatter,其余模式组合仍按原顺序经组大小快捷判断或落到 `NotImplementedError`,非 CP 或 tp_size != attn_cp_size 场景行为不变。

## 关联
- [PR #18233](https://github.com/sgl-project/sglang/pull/18233)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
