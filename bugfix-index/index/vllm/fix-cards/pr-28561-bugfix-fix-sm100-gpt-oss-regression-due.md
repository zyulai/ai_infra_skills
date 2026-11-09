# PR #28561: [Bugfix] Fix SM100 gpt-oss regression due to faulty attn sink support

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28561](https://github.com/vllm-project/vllm/pull/28561) |
| 作者 | mgoin |
| 合并日期 | 2025-11-13 |
| labels | ready, v1, gpt-oss, nvidia |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
gpt-oss 在 Blackwell 上后端探测只剩 Triton;让 FlashInfer TRTLLM 声明支持 attention sink 恢复可选后端。

## 问题现象
Blackwell(SM100)上 `vllm serve openai/gpt-oss-20b --load-format dummy` 启动日志显示 `Valid backends: ['TRITON_ATTN']`、`Using TRITON_ATTN backend.`,即唯一可用注意力后端是 Triton。回归由 #24794 引入。gsm8k 评测对比:main accuracy 0.299 / invalid 0.167;本 PR accuracy 0.310 / invalid 0.162(延迟 36.881s、吞吐 8244.621 output tokens/s)。

## 根因
正文:gpt-oss 需要 attention sink 支持,#24794 引入的回归使 Blackwell 上只有 Triton 满足该要求;正文指出 "FlashInfer can support attention sinks through the TRTLLM backend",但后端能力探测中 FlashInfer 未声明 sink 支持,导致其被排除。

## 修复方案
- `vllm/v1/attention/backends/flashinfer.py`:FlashInfer 后端新增 `supports_sink()` 类方法——在 `force_use_trtllm_attention() is False`(如显式 `VLLM_USE_TRTLLM_ATTENTION=0`)时返回 False,否则返回 `supports_trtllm_attention()`(SM100 + cubin 可用);
- `vllm/utils/flashinfer.py`:新增带缓存的 `has_flashinfer_cubin()`(检查 env 或 `flashinfer_cubin` 包),`has_flashinfer`/`has_nvidia_artifactory` 改用它;批不变模式对 TRTLLM 的禁用从 `force_use_trtllm_attention` 移到 `supports_trtllm_attention` 内。

```diff
+    @classmethod
+    def supports_sink(cls) -> bool:
+        """FlashInfer supports sinks when TRTLLM attention is available (SM100)."""
+        from vllm.utils.flashinfer import (
+            force_use_trtllm_attention,
+            supports_trtllm_attention,
+        )
+
+        # Respect explicit disable flag (e.g., VLLM_USE_TRTLLM_ATTENTION=0)
+        if force_use_trtllm_attention() is False:
+            return False
+
+        # Check if TRTLLM is supported on this platform
+        return supports_trtllm_attention()
```

## 影响与回归风险
- 影响窗口:正文未写(回归由 #24794 引入)。
- 回归面:FlashInfer 后端能力声明(sink)与 TRTLLM 注意力可用性/强制判定;批不变模式(VLLM_USE_TRTLLM_ATTENTION 与 batch-invariant 组合)的行为路径变化;flashinfer-cubin 探测统一。

## 关联
- #24794(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
