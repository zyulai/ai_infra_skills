# PR #21391: Fix Kimi K2.5 dp attention+ spec decoding launch crash

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21391](https://github.com/sgl-project/sglang/pull/21391) |
| 作者 | Qiaolin-Yu |
| 合并日期 | 2026-03-26 |
| labels | high priority, run-ci |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
Kimi K2.5 开 DP attention + spec decoding 且带多模态输入时,draft embedding 无法处理 mm input 导致越界崩溃;按 qwen3_5_mtp 的既有模式修正 mm input embedding。

## 问题现象
正文描述:当存在多模态(mm)输入时,draft embedding 无法处理它,会出现 out of boundary(越界)问题,表现为 Kimi K2.5 在 dp attention + spec decoding 组合下的启动/运行崩溃(见标题)。对应 issue #21336。

## 根因
PR 正文披露:draft embedding 不能处理 mm input,会产生越界问题。(由 diff 推断)`llama_eagle3.py` 的 `forward` 在 `input_embeds is None` 时一律执行 `self.embed_tokens(input_ids)`,完全忽略 `forward_batch.mm_input_embeds`;对 mm 输入,`input_ids` 与 embeds 的对齐关系不再成立,按 ids 查 embedding 造成越界。正文指出修法与 `qwen3_5_mtp.py` L129-L131 的既有模式一致。

## 修复方案
`python/sglang/srt/models/llama_eagle3.py` 的 `forward`:
```diff
         if input_embeds is None:
-            embeds = self.embed_tokens(input_ids)
+            embeds = forward_batch.mm_input_embeds
+            if (
+                forward_batch.forward_mode.is_extend()
+                and forward_batch.contains_mm_inputs()
+                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
+            ):
+                assert embeds is not None
+                embeds = torch.cat(
+                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]
+                )
+            if embeds is None:
+                embeds = self.embed_tokens(input_ids)
```
即 mm input 时优先使用 `forward_batch.mm_input_embeds`,extend 且含 mm input 且非 draft_extend 时,仅对最后一个 token 用 `embed_tokens` 重新计算;无 mm input 时回落到原逻辑。另在 `test/registered/8-gpu-models/test_kimi_k25.py` 新增 `TP8+DP8+MTP` 测试变体(`--dp=8 --enable-dp-attention` + eagle3 参数),并移除 `--mem-frac=0.85`。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:llama_eagle3 draft 模型的 embeds 计算路径。非 mm 输入行为不变(回落 `embed_tokens(input_ids)`);mm 输入在 extend/draft 模式下的 embeds 构造方式改变。新增 8-GPU 测试变体会进入 CI 排布。
- 正文验证:gsm8k 0.936;step=3、topk=1、draft_num_token=4 配置下 accepted length 稳定 > 2.5。

## 关联
- [#21336](https://github.com/sgl-project/sglang/issues/21336)(本 PR 关闭该 issue)
- [qwen3_5_mtp.py 中的同型修法](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5_mtp.py#L129-L131)(正文引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
