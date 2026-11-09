# PR #62234: [Bugfix][Spec Decode] Align speculator hidden norm in GLM-4.5/4.6 MoE and GLM-OCR MTP with GLM5Next to recycle post-final-hidden.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @spandantiwari |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding |
| Label | bug, speculative-decoding, ready, ci/build, glm |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/62234 |

## 一句话结论

修正 GLM 系三个 MTP drafter 回收 pre-final-norm hidden 的契约偏差,对齐 glm5next 的 post-norm recycle,acceptance 实测无回归。

## 动机

`glm4_moe_mtp.py`(GLM-4.5 / 4.5-Air / 4.6)、`glm4_moe_lite_mtp.py`(GLM-4.7-Flash)、`glm_ocr_mtp.py`(GLM-OCR)三个 MTP 层的 forward 以 `hidden_states = residual + hidden_states` 收尾,返回的是 pre-final-norm hidden;而 speculator 会把模型返回的 hidden 存为下一步的 `previous_hidden_states` 喂进 `hnorm`,导致下一 draft 步输入的 hidden 少了一次最终 RMSNorm。GLM/DeepSeek-NextN 的既有契约(已由 #47448 落地,SGLang 的 `glm4_moe_nextn`/`deepseek_nextn` 同)是回收 post-final-norm hidden。另注:#62986 只修了 `Glm4MoeMTP`,本 PR 覆盖全部三个模型并包含 V1 注册。

## 优化点

三个 MTP 层统一改为融合 add+norm:`hidden_states, _ = self.shared_head.norm(hidden_states, residual=residual)` 后返回 `(hidden_states, hidden_states)`——两个元组位置都是 post-norm 张量(与 #47448/glm5next 一致);`compute_logits` 相应去掉第二次 `shared_head` 归一化,直接消费 post-norm hidden,logits 输入不变。同时把 `Glm4MoeMTPModel`、`Glm4MoeLiteMTPModel`、`GlmOcrMTPModel` 注册进 `SpecDecodeBaseProposer.model_returns_tuple()`,否则 Model Runner V1 在首个 draft 步就会因无法解包 tuple 崩溃(MRV2 本就通用解包)。影响面:单模块 AR 路径 `num_nextn_predict_layers=1` 且 k>=2(GLM-4.5/4.6 常见情形)与多模块路径;k=1 不受影响。

## 关键改动

```text
vllm/model_executor/models/glm4_moe_mtp.py / glm4_moe_lite_mtp.py / glm_ocr_mtp.py: forward 返回 post-norm tuple,compute_logits 去掉二次 norm
vllm/v1/spec_decode/llm_base_proposer.py: model_returns_tuple() 注册三个 GLM 架构
tests/v1/spec_decode/test_mtp.py: 新增 test_glm_mtp_feeds_back_final_normed_hidden
```

```python
hidden_states, _ = self.shared_head.norm(hidden_states, residual=residual)
return hidden_states, hidden_states
```

## 性能声明

PR 明确"无性能回归,也无性能声明"。配对 acceptance 实测:GLM-4.5-Air bf16、MI355X、method=mtp k=3、600 条 gsm8k 固定前缀逐条配对——p0 0.7330→0.7298(p=0.88)、p1 0.5438→0.5479(p=0.86)、p2 0.2232→0.2277(p=0.82),配对每 prompt accepted tokens +0.018±0.016(Wilcoxon p=0.25),均无统计显著变化。e2e 逐位 acceptance(GLM-OCR,MI355X,spec tokens=2):MRV2 [0.610, 0.285],MRV1 [0.592, 0.271]。

## 验证与回归风险

新单测用真实 `get_model`(dummy weights)断言 feedback hidden 等于 forward hook 捕获的 mtp_block 输出过 `shared_head.norm`(PR 基线上 3 个全 fail,本 PR 后 3 passed);新 e2e `tests/v1/e2e/spec_decode/mtp/glm/test_mtp.py` 在两个 model runner 上断言两步 acceptance 阈值(>0.4 / >0.15),Buildkite 步骤扩展为 "DeepSeek + MiMo + GLM"。speculative decoding 逐 token 验证保证目标侧输出无损;风险主要是 V1/MRV2 两个 runner 的 tuple 解包路径与后续 GLM MTP 变体需沿用同一契约。

## 关联

前序:#47448(DeepSeek/GLM NextN post-norm recycle 契约)、#62986(仅 Glm4MoeMTP 的同修复,可被本 PR 取代)、#62987(Glm4MoeMTP torch.compile,独立)。参照 SGLang `glm4_moe_nextn.py`/`deepseek_nextn.py`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
