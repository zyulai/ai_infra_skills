# PR #39029: nano_nemotron_vl: fix tensor device mismatch exception when video profiling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39029](https://github.com/vllm-project/vllm/pull/39029) |
| 作者 | netanel-haber |
| 合并日期 | 2026-04-05 |
| labels | ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
nano_nemotron_vl 视频路径中 ID 张量默认建在 CPU 与 GPU 嵌入跨设备冲突,显式指定 device 修复。

## 问题现象
标题披露:nano_nemotron_vl 模型在处理视频(video profiling,即 `_create_final_video_embeddings` 所在的视频嵌入生成路径)时抛出张量设备不匹配(device mismatch)异常。正文未描述具体现象、报错日志与影响面——PR 正文为空,仅有 diff。

## 根因
正文未披露根因(正文为空)。(由 diff 推断)`video_embeddings` 位于 GPU 上,而 `repl_token_ids = torch.tensor(video_repl.full)` 与 `embed_token_ids = torch.tensor(self._img_context_token_ids)` 两个张量未指定 device,默认创建在 CPU;随后 `torch.isin(repl_token_ids, embed_token_ids)` 及据此生成的视频位置 mask 需要与 GPU 上的嵌入比较,触发跨设备运算异常。

## 修复方案
修改 `vllm/model_executor/models/nano_nemotron_vl.py` 的 `_create_final_video_embeddings`:取 `device = video_embeddings.device`,两个 `torch.tensor(...)` 调用都显式传入 `device=device`:

```diff
+        device = video_embeddings.device
         # video_repl.full is a list of token IDs
-        repl_token_ids = torch.tensor(video_repl.full)
+        repl_token_ids = torch.tensor(video_repl.full, device=device)
         # Get embedding token IDs for image context (use pre-tokenized version)
-        embed_token_ids = torch.tensor(self._img_context_token_ids)
+        embed_token_ids = torch.tensor(self._img_context_token_ids, device=device)
```

## 影响与回归风险
- 影响窗口:正文未说明;可确认该问题存在于引入 `torch.isin` 视频嵌入逻辑之后、本修复之前的提交范围(具体提交正文未给出)。
- 回归面:仅 `nano_nemotron_vl.py` 的 `_create_final_video_embeddings` 一个函数;将两个小张量显式放到嵌入所在设备,语义不变,仅消除隐式 CPU 假设。在单进程单设备场景下无副作用;若上层假设这些张量留在 CPU(未见此类用法),需留意。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
