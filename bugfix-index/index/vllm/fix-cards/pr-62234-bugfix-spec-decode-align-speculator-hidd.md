# PR #62234: [Bugfix][Spec Decode] Align speculator hidden norm in GLM-4.5/4.6 MoE and GLM-OCR MTP with GLM5Next to recycle post-final-hidden.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62234](https://github.com/vllm-project/vllm/pull/62234) |
| 作者 | spandantiwari |
| 合并日期 | 2026-11-04 |
| labels | bug, speculative-decoding, ready, ci/build, glm |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
三个 GLM MTP 模型回收 pre-final-norm 隐状态做草稿反馈,缺一次 RMSNorm,改为应用最终 norm 并返回 tuple、注册 V1 解包修复。

## 问题现象
三个 GLM 系 MTP 模型——`glm4_moe_mtp.py`(GLM-4.5 / GLM-4.5-Air / GLM-4.6)、`glm4_moe_lite_mtp.py`(GLM-4.7-Flash)、`glm_ocr_mtp.py`(GLM-OCR)——从 MTP 层返回 final-norm 之前的隐状态,speculator 把这个少了一次 RMSNorm 的隐状态回收到下一草稿步的 `hnorm` 输入,与既定 GLM/DeepSeek-NextN 契约(#47448 及 SGLang `glm4_moe_nextn`/`deepseek_nextn` 的 glm5next 实现)不符。另外 Model Runner V1 的 `SpecDecodeBaseProposer` 只解包已注册架构的 tuple 返回,不注册则 V1 在第一草稿步即崩:`TypeError: only integer tensors of a single element can be converted to an index`,进而 `EngineDeadError`(PR 正文验证小节记录)。影响单模块 AR 路径(`num_nextn_predict_layers=1`,k≥2 的常见 GLM-4.5/4.6 情形)与多模块路径;k=1 不受影响。

## 根因
正文披露的根因:MTP 层 forward 以 `hidden_states = residual + hidden_states; return hidden_states` 结束,返回的是 pre-final-norm 隐状态,而 `compute_logits` 会再对它应用 `shared_head`;logits 路径一直是单次 norm 的正确输入,但 speculator 存储模型返回的 hidden 作为下一步的 `previous_hidden_states`(送入 `hnorm`),于是回收的状态缺少 DeepSeek/`glm5next` 都会应用的最终 `shared_head` RMSNorm。

## 修复方案
按文件改三个 MTP 模型(`glm4_moe_mtp.py`、`glm4_moe_lite_mtp.py`、`glm_ocr_mtp.py`):
1. forward 改为应用一次融合 add+norm 并回收 post-final-norm 隐状态,两个 tuple 位置返回同一张量。
2. `compute_logits` 直接消费该隐状态(`shared_head.head, hidden_states`),去掉第二次 norm。
3. 在 `vllm/v1/spec_decode/llm_base_proposer.py` 的 `model_returns_tuple()` 注册 `Glm4MoeMTPModel`、`Glm4MoeLiteMTPModel`、`GlmOcrMTPModel`,使 Model Runner V1 正确解包 tuple。
另新增单测(真实对象跑 forward,断言 feedback hidden 等于 `shared_head.norm` 输出)与 MRV1/MRV2 e2e 测试,并扩展 Buildkite CI 步骤。

关键 diff 摘录:
```diff
-        hidden_states = residual + hidden_states
-        return hidden_states
+        hidden_states, _ = self.shared_head.norm(hidden_states, residual=residual)
+        return hidden_states, hidden_states
...
-        logits = self.logits_processor(
-            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
-        )
+        logits = self.logits_processor(mtp_layer.shared_head.head, hidden_states)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;影响 k≥2 的 GLM-4.5/4.5-Air/4.6、GLM-4.7-Flash、GLM-OCR MTP 投机解码。
- 回归面:改动为 draft-only,投机解码会验证每个草稿 token,目标模型可见输出不变;仅草稿器回收状态变化(影响第 ≥1 步草稿)。作者给出 GLM-4.5-Air 上 k=3、600 条 gsm8k 固定前缀配对测量:各位置与整体接受率无统计显著差异(Wilcoxon p=0.25)。回归面集中在:三个 MTP 文件的 forward/compute_logits、`model_returns_tuple()` 注册集合、以及 Model Runner V1 对 tuple 的解包路径。

## 关联
- [#47448](https://github.com/vllm-project/vllm/pull/47448)(已合并的 glm5next/DeepSeek 契约先例)
- [#62986](https://github.com/vllm-project/vllm/pull/62986)(仅修 Glm4MoeMTP 的后续 PR,描述中提出在 V1 注册补齐后关闭,现已包含)
- [#62987](https://github.com/vllm-project/vllm/pull/62987)(为 Glm4MoeMTP 启用 torch.compile,独立)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
