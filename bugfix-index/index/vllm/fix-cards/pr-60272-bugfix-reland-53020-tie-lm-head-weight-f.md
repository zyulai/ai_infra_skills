# PR #60272: [Bugfix] Reland #53020: tie lm_head.weight for Nemotron Parse when checkpoint omits it

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60272](https://github.com/vllm-project/vllm/pull/60272) |
| 作者 | aniskumar-nv |
| 合并日期 | 2026-10-08 |
| labels | bug, ready, multi-modality |
| 热度 | 👍 0 · 💬 27 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
Nemotron Parse 2.0 checkpoint 无 lm_head.weight,输出头保持随机初始化返回乱码;按 tie_word_embeddings 绑定修复。

## 问题现象
`NemotronParseForConditionalGeneration` 构建独立 `ParallelLMHead` 且只从 `lm_head.*` 权重填充。checkpoint 设置 `tie_word_embeddings` 且不携带 `lm_head.weight`(如 `nvidia/NVIDIA-Nemotron-Parse-2.0`)时,输出头保持随机初始化——服务正常启动、报告健康,但输出是乱码。实测:v0.30.0 未打补丁时 greedy 请求输出 50 个重复 `<s>`;打补丁后输出正确的表格 token 序列,且与 HF 仓库自带 workaround 的输出逐字节一致。

## 根因
正文披露:该类模型只按 `lm_head.*` 权重初始化输出头,从不处理 `tie_word_embeddings` 场景;省略 `lm_head.weight` 的 checkpoint(2.0:该标志 true/ttrue,无 lm_head.weight,666 个 tensors;对比 v1.2:false,有该权重,667 个 tensors)导致输出头随机初始化。本 PR 是重新落地 #53020:此前 #59805 回滚并非修复本身有误,而是 CI 的 `tokenizers==0.23.1` pin 在加载 2.0 tokenizer 时抛 `DuplicatePatternError`(NFKC 归一化后 14 对 added tokens 重复),#59796 已 bump 到 0.23.2 解除阻塞。

## 修复方案
`vllm/model_executor/models/nemotron_parse.py` `__init__`:读取 `config.tie_word_embeddings` 或 `config.decoder.tie_word_embeddings`,为真时调用 `self.lm_head.tie_weights(self.decoder.embed_tokens)`(经由 `quant_method.tie_weights` 路由,而非直接赋值 `.weight`)。

关键 diff 摘录:
```diff
+        tie_word_embeddings = bool(
+            getattr(config, "tie_word_embeddings", False)
+            or getattr(config.decoder, "tie_word_embeddings", False)
+        )
+        if tie_word_embeddings:
+            self.lm_head = self.lm_head.tie_weights(self.decoder.embed_tokens)
```
测试:恢复 `tests/models/multimodal/generation/test_nemotron_parse.py` 中 `NVIDIA-Nemotron-Parse-2.0` 参数化(唯一覆盖 tied-head 路径的用例)。diff 与被回滚的 commit `83cadd65d9` 逐字节一致(`git patch-id` 匹配)。

## 影响与回归风险
- 影响窗口:#53020 首次合并(2026-10-01)至 #59805 回滚(2026-10-02)期间曾存在,回滚后窗口消失,本 PR 恢复修复。
- 回归面:仅 Nemotron Parse 模型加载;v1.2 checkpoint 的 `tie_word_embeddings` 为 false,不受影响(正文用两 checkpoint 的配置差异说明);其他设 `tie_word_embeddings` 的 checkpoint 加载路径会新增绑定行为。

## 关联
- 原始 bug 报告 [#53019](https://github.com/vllm-project/vllm/issues/53019)
- Re-lands [#53020](https://github.com/vllm-project/vllm/pull/53020)
- 回滚 PR [#59805](https://github.com/vllm-project/vllm/pull/59805)
- 解除阻塞的 pin bump [#59796](https://github.com/vllm-project/vllm/pull/59796)
- 同 diff 的 #60266(被本 PR 取代)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
