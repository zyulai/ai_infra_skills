# PR #25406: [Speculators][Speculative Decoding] Fix gpt-oss eagle3 accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25406](https://github.com/vllm-project/vllm/pull/25406) |
| 作者 | jiahanc |
| 合并日期 | 2025-09-23 |
| labels | bug, gpt-oss, llama, ready, speculative-decoding, v1 |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论

gpt-oss 多注意力组下 EAGLE 只取第 1 个 attention builder,选错注意力组且覆写目标模型 KV cache,精度崩坏;按 EAGLE 层名定位正确 builder 修复。

## 问题现象

gpt-oss-120b 配 nvidia/gpt-oss-120b-Eagle3 投机解码时精度崩坏:GSM8K(lm_eval,limit 0.8)修复前 flexible-extract 仅 0.0038、strict-match 0.0000;SpecDecoding 平均接受长度仅 1.04,平均 draft 接受率 1.2%(per-position 0.035/0.002/0.000)。此外 nvidia/gpt-oss-120b-Eagle3 配置本身在 vLLM 上跑不起来(vocab_size、lm_head 等 config 问题)。

## 根因

正文披露两点:1) 现有 spec-dec 实现假设只有 1 个 attention group(`gpu_model_runner.py` 中直接取值),`eagle.py` 也只用第 1 个 attention builder;对于 gpt-oss 这类有多个注意力的模型,会选错 attention group,draft 模型因此用错注意力,同时会覆写(overwrite)目标模型的 KV cache,造成精度问题。2) gpt-oss eagle3 的 config 问题,如 vocab_size 与 lm_head。

## 修复方案

1. `vllm/v1/spec_decode/eagle.py`:新增 `_get_attention_metadata_builder()`,遍历 `runner.attn_groups` 找包含 EAGLE 首层(`attn_layer_names[0]`)的组并取其 builder,找不到则断言报错;`propose` 中缓存该 builder 复用,替换原先 `self.runner.attn_groups[0][0].get_metadata_builder()` 的两处硬编码。
2. `vllm/v1/worker/gpu_model_runner.py`:`_prepare_inputs` 中仅当 `drafter.attn_layer_names[0]` 属于当前 `kv_cache_group_spec.layer_names` 时才记录 `spec_decode_common_attn_metadata`(EagleProposer 分支)。
3. `llama_eagle.py`/`llama_eagle3.py`:`draft_vocab_size` 缺失时回退为 base 的 `vocab_size`。
4. `eagle.py load_model`:eagle3 时若 draft 与 target 的 lm_head 形状一致则共享 target 的 lm_head,否则单独加载 draft 的 lm_head(此前 eagle3 一律不共享)。
5. `vllm/config/model.py`:`quant_method` 为空串时归一化为 None。

修复后 GSM8K flexible-extract 0.0038 → 0.8362,平均接受长度 1.04 → 2.81,平均 draft 接受率 1.2% → 60.2%。

## 影响与回归风险

- 影响窗口:正文未说明具体版本;继承自 PR #23596 的工作。
- 回归面:改动覆盖 EAGLE proposer 的 builder 选择、`_prepare_inputs` 的 spec-dec 元数据捕获条件、eagle/eagle3 模型的 vocab 与 lm_head 加载逻辑,以及量化配置归一化;影响所有 EAGLE/EAGLE3 投机解码路径,尤其多 KV cache group 模型。`validate_same_kv_cache_group` 仍在,单注意力组模型的选组结果应与原逻辑一致。测试侧给 proposer 补了 `_get_attention_metadata_builder` 的 mock。

## 关联

- 继承自 https://github.com/vllm-project/vllm/pull/23596
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py#L1178-L1180(正文引用的问题代码)
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py#L194-L199(正文引用的问题代码)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
