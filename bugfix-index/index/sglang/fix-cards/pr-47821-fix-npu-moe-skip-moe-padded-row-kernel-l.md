# PR #47821: [Fix] [NPU] MOE: Skip MoE padded-row kernel launches for empty tensors in case of crashing on NPU

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47821](https://github.com/sgl-project/sglang/pull/47821) |
| 作者 | hanwlax |
| 合并日期 | 2026-11-04 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 10 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论

NPU 上空 TopK 张量触发零尺寸 Triton grid(coreDim=0)崩溃,空张量改为 no-op 跳过 kernel 启动。

## 问题现象

Kimi K3 启动 prefill 且开启 attention DP 时,idle rank 会产生空的 TopK 张量。共享的 padding 辅助函数会提交零尺寸的 Triton grid,NPU 以 coreDim=0 拒绝,报异步错误;该错误可能延迟到 DeepEP 同步阶段才暴露出来,难以直接定位。

## 根因

正文披露:idle rank 产生空 TopK 张量后,共享 padding helper(`_fill_padded_rows`)未对空张量做短路处理,直接以零行数启动 Triton kernel,而 NPU 不接受 coreDim=0 的 grid,从而崩溃。

## 修复方案

在 `python/sglang/kernels/ops/moe/fill_padded_rows.py` 的 `_fill_padded_rows` 中,于启动 padded-row 填充 kernel 之前,把空张量视为 no-op 直接返回;同时保留既有的元数据校验与非空张量行为不变。

关键 diff 摘录:

```diff
     n_rows, n_cols = x.shape
+    if x.numel() == 0:
+        return
     strided = _use_strided(n_rows, n_cols * x.element_size())
     _fill_padded_rows_kernel[(_MAX_PROGRAMS if strided else n_rows,)](
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 `fill_padded_rows.py` 的 `_fill_padded_rows` 入口新增空张量短路;非空张量路径(包括 `_use_strided` 判断与 kernel 参数)完全不变,元数据校验(device 一致性检查)保留在其之前。影响所有经过该 padding helper 的 MoE 路径在空输入下的行为:从"启动零尺寸 grid 报错"变为"无操作直接返回"。正文未提供 accuracy 测试结果(对应 Checklist 项未勾选;CI 状态栏 Base/Extra 为失败、AMD ROCm 10 通过)。

## 关联

正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
