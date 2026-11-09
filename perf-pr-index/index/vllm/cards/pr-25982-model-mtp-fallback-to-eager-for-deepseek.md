# PR #25982: [Model] MTP fallback to eager for DeepSeek v32

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @luccafong |
| 合并 | 2025-10-01 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, v1, deepseek |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 5 个文件：`vllm/config/speculative.py`、`vllm/v1/spec_decode/eagle.py`、`vllm/v1/attention/backends/mla/indexer.py`、`tests/v1/spec_decode/test_eagle.py`、`tests/v1/spec_decode/test_mtp.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/25982 |

## 一句话结论

DeepSeek V3.2 的 MTP draft 在 cudagraph 支持完成前默认回退 eager 执行，用 MTP 保住整体约 25% 的 decode 加速而不被编译问题拖垮。

## 动机

Cudagraph + MTP（DeepSeek V3.2）的支持仍在开发中，当时两者组合不可用；若整个模型因此退回 eager，主模型吞吐损失过大。PR 的目的有二：其一，在 MTP 部分默认启用 eager，使 V3.2 + MTP 可以与 piecewise 编译组合跑通并保持验证过的精度；其二，修复 Eagle 与 MTP 的单元测试（@njhill 提及），使测试与新 metadata builder 结构对齐。

## 优化点

机制上是"细粒度 eager 回退"而非全局回退。`SpeculativeConfig` 新增独立的 `enforce_eager` 字段（可覆盖 model_config 的默认值），在 `__post_init__` 中检测 `model_type == "deepseek_v32"` 时强制置 True（附 FIXME：cudagraph with v32 MTP 未支持，问题修复后移除）。`EagleProposer.use_cuda_graph` 的判定相应加上 `not speculative_config.enforce_eager`，即 draft 模型单独走 eager，target 模型仍可用 piecewise 编译。同时把 DeepseekV32IndexerMetadataBuilder 的 `cudagraph_support` 从 `UNIFORM_BATCH` 收紧为 `UNIFORM_SINGLE_TOKEN_DECODE`（MTP 场景 decode 为单 token），并让 proposer 维护独立的 `draft_indexer_metadata_builder` 与 `indexer_layer_names`， Eagle 测试 mock 的 `get_layers` side_effect 补上 indexer 层两组返回值。

## 关键改动

```text
vllm/config/speculative.py: SpeculativeConfig 增加 enforce_eager；deepseek_v32 + MTP 默认强制 eager
vllm/v1/spec_decode/eagle.py: use_cuda_graph 增加 speculative_config.enforce_eager 判定；新增 draft indexer metadata builder
vllm/v1/attention/backends/mla/indexer.py: V32 indexer cudagraph_support 收紧为 UNIFORM_SINGLE_TOKEN_DECODE
tests/v1/spec_decode/test_eagle.py / test_mtp.py: 适配 indexer 层 mock 与 metadata builders
```

```python
if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
    # FIXME(luccafong): cudgraph with v32 MTP is not supported,
    # remove this when the issue is fixed.
    self.enforce_eager = True
```

## 性能声明

测试条件：`deepseek-ai/DeepSeek-V3.2-Exp`，TP8，`max_model_len=20000`，`max_num_seqs=256`，`speculative_config {"num_speculative_tokens":1, "method":"mtp"}`，piecewise 编译。GSM8K（20-shot，200 并发请求）评测耗时：开 MTP 9:58，不开 MTP 12:08，即 25% speedup（PR 原文措辞）。精度：GSM8K flexible-extract 0.9507、strict-match 0.9500。SpecDecoding metrics：Mean acceptance length 1.92，per-position acceptance rate 0.919，Accepted throughput 80.90 tokens/s。

## 验证与回归风险

测试：`tests/v1/spec_decode/test_eagle.py`（mock `get_layers` 增加 indexer 层返回、metadata builders 挂接）与 `tests/v1/spec_decode/test_mtp.py`（同样适配）；PR 正文给出 vllm serve + lm_eval 的完整复现命令。风险面：该回退是硬编码的模型类型特判（deepseek_v32 + method mtp），cudagraph 路径对该组合暂不可用（等 FIXME 移除）；V32 indexer 的 `UNIFORM_SINGLE_TOKEN_DECODE` 限制只接受单 token decode 的 cudagraph 支持，非 MTP 多 token 场景可能受 `reorder_batch_threshold=1` 之外的约束；其他模型的 MTP/EAGLE 不受影响。

## 关联

同区域：DeepSeek V3.2 MLA indexer 相关优化与 MTP 支持（PR 正文未列具体编号）；修复由 @njhill 协作的 Eagle/MTP 测试。无其他明确关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
