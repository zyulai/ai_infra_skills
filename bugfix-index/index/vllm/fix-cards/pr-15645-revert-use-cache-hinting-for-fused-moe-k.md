# PR #15645: Revert "Use Cache Hinting for fused_moe kernel (#15511)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15645](https://github.com/vllm-project/vllm/pull/15645) |
| 作者 | wrmedford |
| 合并日期 | 2025-03-28 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
fused_moe kernel 的缓存提示改动触发 Triton PTX codegen 错误,revert #15511 恢复普通 tl.load。

## 问题现象
正文描述:此前 "Use Cache Hinting for fused_moe kernel (#15511)" 引入的改动,在后续 CI 中被捕获到 "Internal Triton PTX codegen error"(Triton 内部 PTX 代码生成错误);此外 issue 讨论线程中还提到了其他问题。正文未给出具体的复现步骤、模型或硬件条件。

## 根因
正文披露:#15511 为 fused_moe Triton kernel 的 `tl.load` 添加 `cache_modifier=".cg"` 与 `eviction_policy="evict_last"` 缓存提示后,触发 Triton 内部 PTX codegen 错误(在后续 CI 中暴露),且 issue 线程中还提到其他问题,说明该优化在此场景下不可靠,因此整体 revert。

## 修复方案
整体 revert 提交 7a888271("Use Cache Hinting for fused_moe kernel (#15511)")。改动集中在 `vllm/model_executor/layers/fused_moe/fused_moe.py`:把 `fused_moe_kernel_gptq_awq` 与 `fused_moe_kernel` 两个 kernel 中对权重 `b_ptrs` 的加载恢复为不带缓存提示的普通 `tl.load`。以 `fused_moe_kernel` 为例:

```diff
-        b = tl.load(
-            b_ptrs,
-            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
-            other=0.0,
-            cache_modifier=".cg",
-            eviction_policy="evict_last",
-        )
+        b = tl.load(b_ptrs,
+                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
+                    other=0.0)
```

`fused_moe_kernel_gptq_awq` 中同理,移除 `cache_modifier` 与 `eviction_policy` 两个参数。

## 影响与回归风险
- 影响窗口:自 #15511 合并引入缓存提示起,至本 revert 合并止;正文未给出具体版本号或提交范围。
- 回归面:仅改回两个 fused_moe kernel 的 load 参数,移除缓存提示后行为回到 #15511 之前的状态;原缓存提示期望的访存优化收益会随之消失(相关性能可能回退),但数值与功能路径不受影响。依赖 #15511 性能表现的基准需重新评估。

## 关联
- https://github.com/vllm-project/vllm/issues/15619
- PR #15511(被 revert 的改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
