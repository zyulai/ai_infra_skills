# PR #33077: [BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33077](https://github.com/vllm-project/vllm/pull/33077) |
| 作者 | JartX |
| 合并日期 | 2026-02-01 |
| labels | bug, qwen, ready, rocm |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Model-specific Fast Path |
| bug 类型 | startup |

## 一句话结论
ROCm 上 Qwen3-Omni 启动 profiling 时 GPU 端 repeat_interleave 触发 hipErrorIllegalState;捕获 RuntimeError 后改用 searchsorted 向量化实现计算 cu_seqlens。

## 问题现象
AMD(ROCm)硬件上初始化 Qwen3-Omni 时发生致命崩溃(worker crash):在 memory profiling 阶段(`determine_available_memory` → `profile_run` → `embed_multimodal` → `self.visual(...)`),`qwen3_omni_moe_thinker.py` forward 中计算 cu_seqlens 的 `torch.repeat_interleave` 在 GPU 上触发 HIP 错误,报错为 `torch.AcceleratorError: HIP error: the operation cannot be performed in the present state`(hipErrorIllegalState)。模型因此无法完成启动,推理不可用。正文附完整 Worker_TP1 崩溃日志。

## 根因
正文披露:这看起来是当前 ROCm driver/backend 在处理视觉 grid 维度时,对动态张量形状操作的稳定性问题(profile_run 阶段 GPU 上的 `torch.repeat_interleave` 触发 hipErrorIllegalState)。diff 注释进一步指明这是 RDNA3(gfx11)特定 bug 的 workaround。

## 修复方案
正文 Fix 一节描述为「把 cu_seqlens 计算(repeat_interleave 与 cumsum)移到 CPU,结果再移回设备」;实际 diff 的实现是在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py` 的 `forward` 中:
- 先按原样尝试 GPU 上的 `torch.repeat_interleave(...).cumsum(...)` 并 `F.pad`;
- 捕获 `RuntimeError` 后打 warning「torch.repeat_interleave not executable, switching to vectorized searchsorted implementation」,改用向量化 fallback:`grid_thw[:,0]` 做 cumsum,用 `torch.searchsorted` 展开重复索引,再对 `values[indices]` 做 cumsum 并 pad,得到等价 cu_seqlens。

```diff
+        try:
+            cu_seqlens = torch.repeat_interleave(
+                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
+            ).cumsum(...)
+            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
+        except RuntimeError:
+            logger.warning(...)
+            repeat_cumsum = repeat_counts.cumsum(0)
+            indices = torch.searchsorted(repeat_cumsum, torch.arange(total_items, ...), right=True)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:Qwen3-Omni(VisionTransformer)forward 的 cu_seqlens 计算路径;正常环境下走原 GPU repeat_interleave 路径,行为不变;触发 RuntimeError 的环境走 searchsorted fallback(结果等价,多一次 warning 日志与少量额外算子);fallback 首次仍可能在触发处抛错一次后被捕获(try/except 语义),属可接受的启动开销。

## 关联
正文未列出(仅 @ 提及 tjtanaa / andrew / mgoin / Lucaskabela)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
