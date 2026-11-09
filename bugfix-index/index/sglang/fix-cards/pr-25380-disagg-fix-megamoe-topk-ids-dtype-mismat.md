# PR #25380: [Disagg] Fix MegaMoE topk_ids dtype mismatch and FakeKVManager missing kv_args

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25380](https://github.com/sgl-project/sglang/pull/25380) |
| 作者 | JoeLee314 |
| 合并日期 | 2026-05-16 |
| labels | (none) |
| 热度 | 👍 2 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
DeepSeek-V4-Pro MegaMoE + fake disaggregation 初始化崩溃:topk_ids 被静默变 int64、FakeKVManager 缺 kv_args;两处一行修复。

## 问题现象
以 `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` 开启 MegaMoE、decode 分离模式(`--disaggregation-transfer-backend fake --ep-dispatch-algorithm fake`)服务 DeepSeek-V4-Pro 时,服务器初始化即崩溃,报两个独立错误:Bug 1 为 `tvm.error.InternalError: ... dtype=int64 ... not in the allowed options: [int32]`(位于 `mega_moe_pre_dispatch.cuh:150`);Bug 2 为 `AttributeError: 'FakeKVManager' object has no attribute 'kv_args'`。作者给出完整启动命令(tp 8 / dp 8 / dp-attention / deepep a2a 等)。

## 根因
正文披露了两个根因及完整分析:
1. `-ep-dispatch-algorithm fake` 时 `topk_output.topk_ids` 变成 `int64`,而 JIT 编译的 `mega_moe_pre_dispatch` 只接受 `int32`。链路:`_topk_ids_logical_to_physical_dynamic()` 用 `partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]` 索引,该 map 经 `torch.tensor()` 创建、默认 `int64`;PyTorch 索引结果 dtype 跟随源张量而非索引张量,`int64_tensor[int32_idx]` 返回 `int64`,静默撤销此前 `.to(torch.int32)` 的转换,int64 随即流入只收 int32 的内核触发 TVM 检查错误(不设该参数时 `init_new()` 返回 None、原样返回故无此问题)。
2. `FakeKVManager` 继承 `BaseKVManager` 但不像 `CommonKVManager` 那样把 `args` 存为 `self.kv_args`,而 decode 事件循环在预分配阶段访问 `self.kv_manager.kv_args.state_types`,运行时崩溃。

## 修复方案
- `python/sglang/srt/layers/moe/mega_moe.py` 的 `_run_mega_routed`:传入内核前显式转换 dtype(与 else 分支已建的 int32 空张量保持一致):
```diff
-        topk_ids_in = topk_ids
-        topk_weights_in = topk_weights
+        topk_ids_in = topk_ids.to(torch.int32)
+        topk_weights_in = topk_weights.to(torch.float32)
```
作者解释不在 `expert_location_dispatch.py` 修 map dtype,而是修在 MegaMoE 入口:内核契约在自己边界强制、多条 topk 路径(grouped/biased/fused/custom)都汇于此、且避免影响非 MegaMoE 路径。
- `python/sglang/srt/disaggregation/fake/conn.py` 的 `FakeKVManager.__init__` 增加 `self.kv_args = args`,对齐 `CommonKVManager` 模式。
- 验证:修复后在 8× NVIDIA B200 上用上述命令成功启动并打印 `The server is fired up and ready to roll!`;准确性/速度测试标为不适用(int64→int32 保值转换,expert 索引很小;`.to(torch.int32)` 在已符合条件时为 no-op)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:MegaMoE 路由入口 `_run_mega_routed` 对所有 topk 输入强制 int32/float32(值域小的 expert 索引安全);fake 传输后端的 KV 管理器新增属性引用,仅为补齐运行时访问,不影响真实传输后端。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
