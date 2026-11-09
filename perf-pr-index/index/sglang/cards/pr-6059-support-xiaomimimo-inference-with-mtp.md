# PR #6059: Support XiaomiMiMo inference with mtp

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ryang-max |
| 合并 | 2025-05-22 |
| 分类 | Speculative Decoding |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 6 个文件(2 新增 1 重命名 3 修改) |
| 链接 | https://github.com/sgl-project/sglang/pull/6059 |

## 一句话结论

为 XiaomiMiMo 接入 MTP 投机解码,单请求解码速度 79.35 → 103.04 token/s,gsm8k 吞吐约 +30% 且精度基本无损。

## 动机

支持 XiaomiMiMo(MiMo-7B-RL)模型推理,并让其能用上 MTP(Multi-Token Prediction)权重做投机解码。实现上复用 SGLang 既有的 EAGLE 投机解码框架与 DeepSeek NextN 的 MTP 模式,而非新造一套投机机制。

## 优化点

新增 `mimo_mtp.py` 作为 draft 模型(改编自 vllm PR #17433 与 `deepseek_nextn.py`):`MiMoMultiTokenPredictorLayer` 由 embed_tokens、token/hidden 两个 RMSNorm、`input_proj`(cat(hidden_layernorm(spec_info.hidden_states), token_layernorm(embed)))、一个 Qwen2DecoderLayer 充当的 `mtp_block` 与 final_layernorm 组成;`MiMoMTP` 对外暴露 logits,并通过 `set_embed_and_head` 与 target 模型共享 embedding 和 lm_head。接入侧:`model_config.py` 在 is_draft_model 且架构为 `MiMoForCausalLM` 时改写为 `MiMoMTP`;`model_runner.py` 修复 draft worker 的 KV cache cell size 计算——此前仅 MLA 分支按 `num_nextn_predict_layers` 取层数,现统一改为 draft worker 用 `getattr(hf_config, "num_nextn_predict_layers", num_effective_layers)`,使 MHA 类 draft 模型(MiMo)也能正确估算显存。用法:`--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`。

## 关键改动

```text
python/sglang/srt/models/mimo_mtp.py: 新增 MiMoMTP draft 模型(MTP layer + 权重映射)
python/sglang/srt/configs/model_config.py: draft 架构名 MiMoForCausalLM → MiMoMTP
python/sglang/srt/model_executor/model_runner.py: draft worker num_layers 的取值修复
python/sglang/srt/models/xiaomi_mimo.py → mimo.py: 重命名
test/srt/models/test_mtp_models.py: 新增 TestMiMoMTP gsm8k 测试
docs/backend/speculative_decoding.ipynb: MTP 使用文档
```

```python
hidden_states = self.input_proj(torch.cat((
    self.hidden_layernorm(forward_batch.spec_info.hidden_states),
    self.token_layernorm(hidden_states),
), dim=-1))
hidden_states, residual = self.mtp_block(...)
```

## 性能声明

- test/send_one(单请求):w/ mtp acc_length=1.76、speed=103.04 token/s;w/o mtp acc_length=1.00、speed=79.35 token/s
- gsm8k(`bench_sglang.py --num-questions 300 --parallel 4`):w/ mtp accuracy 0.780、latency 145.666s、output throughput 377.616 token/s;w/o mtp accuracy 0.787、latency 193.606s、throughput 291.742 token/s
- 作者结论:"Throughput increased by about 30% without accuracy loss"(0.780 vs 0.787 略降)。模型 XiaomiMiMo/MiMo-7B-RL;硬件未注明

## 验证与回归风险

新增 `test/srt/models/test_mtp_models.py`(启动 MiMo-7B-RL + EAGLE MTP 服务,跑 gsm8k 200 题 parallel 128,断言 accuracy > 0.7)。风险:`profile_max_num_token` 的 draft worker 层数逻辑改动影响所有投机解码 draft 模型的 KV cache 显存估算;架构名映射仅覆盖 `MiMoForCausalLM`;权重加载依赖 checkpoint 的 `model.mtp_layers.<n>.` 命名前缀映射。

## 关联

前序/参照:DeepSeek NextN MTP(`deepseek_nextn.py`)、vllm PR #17433、SGLang EAGLE 投机解码框架(编号素材未注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
