# PR #50693: Fix DSpark warmup without sparse index buffer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50693](https://github.com/vllm-project/vllm/pull/50693) |
| 作者 | xijiaat |
| 合并日期 | 2026-08-10 |
| labels | ready |
| 热度 | 👍 5 · 💬 8 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
DSpark 启动 warmup 在 SWA-only 层先断言 topk 缓冲存在致启动失败;把 swa_only 分支提前、断言移入 sparse 路径。

## 问题现象
DSpark 启动时 `profile_run` 调用 `forward_mqa` 且无 attention metadata(用于确定 warmup 工作区大小);SWA-only draft 层(`compress_ratio <= 1`)刻意不分配 `topk_indices_buffer`,而 warmup 路径在把 `top_k` 置零前先断言缓冲存在,引擎在加载完整 target 与 draft 模型后失败:`flashmla.py:99: assert self.topk_indices_buffer is not None` → `AssertionError`,API server 永不健康。复现环境:1 x NVIDIA B300(275,040 MiB,sm_100f,TP1)、nightly 镜像 `vllm/vllm-openai:nightly-0f17394564fa2fccd332cf63321314884c15ee37`、模型 deepseek-ai/DeepSeek-V4-Flash-0731(revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062),带 `--speculative-config '{"method":"dspark","num_speculative_tokens":7}'` 等配置。修复后同一部署完成 profiling/KV-cache/内核 warmup/API 启动(引擎初始化 420.82 s),3/3 请求 HTTP 200,DSpark 计数器确认投机解码真实执行(drafts 20 / draft tokens 140 / accepted 51)。

## 根因
正文:warmup 路径的断言顺序错误——SWA-only(`compress_ratio <= 1`)本就不需要 sparse index buffer,但代码在处理 `swa_only` 之前就断言 `topk_indices_buffer` 非 None(Issue #50615 中由 @fank 给出复现与根因分析,本 PR 实现其建议的最小修复)。

## 修复方案
改 `vllm/models/deepseek_v4/nvidia/flashmla.py` 的 `forward_mqa`:先处理 SWA-only 情形(`top_k = 0`,不要求 sparse index buffer),`assert self.topk_indices_buffer is not None` 移入需要缓冲的 sparse 分支。最终 diff 仅含该生产源文件(专用回归测试按审稿意见移除)。

```diff
-            assert self.topk_indices_buffer is not None
-            top_k = 0 if swa_only else self.topk_indices_buffer.shape[-1]
+            if swa_only:
+                top_k = 0
+            else:
+                assert self.topk_indices_buffer is not None
+                top_k = self.topk_indices_buffer.shape[-1]
```

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:仅 `deepseek_v4/nvidia/flashmla.py` 的 `forward_mqa` 中 SWA-only warmup 与 sparse 路径的 `top_k` 取值;sparse 路径(需缓冲)的断言保持不变。

## 关联
- Issue #50615
- #41834(邻近的 SM12x enablement 分支,经查其当前 head 仍含同样的 SWA-only 缓冲假设)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
