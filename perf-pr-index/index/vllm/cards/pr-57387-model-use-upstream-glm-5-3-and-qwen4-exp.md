# PR #57387: [Model] Use upstream GLM-5.3 and Qwen4-Exp configs and processor

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hmellor |
| 合并 | 2026-10-02 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | speculative-decoding, ready, ci/build, multi-modality, qwen, glm |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 23 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/57387 |

## 一句话结论

移除 GLM-5.3/Qwen4-Exp 的 vendored 配置改用 upstream，净删 1961 行且行为等价。

## 动机

Transformers 自 v5.16.1 起内置 `glm5_next`、自 v5.16.0 起内置 `qwen4_exp`，vLLM 中 vendored 的副本可以删除。PR 把 `requirements/common.txt` 中 transformers 下限从 `>= 5.10.4` 提到 `>= 5.16.1`，仅让声明的最小版本与 CI 已精确 pin 的 `transformers==5.16.1` 对齐。upstream 类并非 drop-in 替换，属性拼写不同，模型代码需同步迁移。

## 优化点

删除 4 个 vendored 文件（glm5_next.py 420 行、glm5next.py 953 行、qwen4_exp 两个 config.py），净 -1961 行。按重命名表迁移若干属性拼写（如 `mla_nope`→`mla_use_nope`、`moe_renormalize`→`norm_topk_prob` 等）；`is_moe`/`is_kda_layer`/`is_linear_attn` 改由 `layer_types` 推导，text 级读取走 `hf_text_config`。保留三处刻意差异（vision `rms_norm_eps` 固定 1e-6、frame sampler 移至 `Glm5NextVideoBackend`、guards 移入 `_validate_supported_config`）。修复 hybrid layer accounting：`get_num_layers_by_block_type` 把稀疏注意力层计入 attention，否则两模型统计为 0 个 attention 层；Qwen4Exp 无条件构建 MoE（删死分支 -38/+6）。

## 关键改动

```text
vllm/config/model.py: 新增 _ATTENTION_LAYER_TYPES，让稀疏注意力层计入 attention 层统计
vllm/models/glm5next/common/model.py: 改用 upstream config 拼写、_validate_supported_config 与 hf_text_config
vllm/config/speculative.py: num_nextn_predict_layers 改走 get_text_config()
requirements/common.txt: transformers 下限升到 >= 5.16.1
（删除）vllm/transformers_utils/configs/glm5_next.py 等 4 个 vendored 文件
```

```diff
+_ATTENTION_LAYER_TYPES = frozenset(
+    {
+        "full_attention",
+        "deepseek_sparse_attention",
+        "qwen_sparse_attention",
+    }
+)
...
-                        t == "full_attention" for t in layer_types_value[start:end]
+                        t in _ATTENTION_LAYER_TYPES
+                        for t in layer_types_value[start:end]
```

## 性能声明

PR 正文未给出定量性能数据。正文列出的数字均为行为等价验证而非性能基准：smart_resize 差分 8232/8232 一致、frame sampler 在 38/53 用例帧索引不同、属性解析 37/39。

## 验证与回归风险

测试计划为 pytest tests/models/glm5next/test_config.py、tests/models/qwen4_exp/test_config.py、tests/transformers_utils/test_config.py、tests/multimodal/test_video.py、tests/test_config.py 及 pre-commit。PR 在 branch 与 base 双 worktree 运行，失败集一致（base 与 branch 均 24 failed/18 skipped，passed 由 433→430，因删 7 用例加 4 用例），pre-commit 含 mypy 通过。风险：作者注明无 GPU、未跑模型 eval；vision `rms_norm_eps` 路径若回归会静默失败；frame sampler 保留 vLLM 自有实现但缺视频 eval；删除 Qwen4Exp 的 dense 分支后未来设置 `mlp_only_layers`/`decoder_sparse_step` 的 checkpoint 不再生效。

## 关联

PR 正文提到 Qwen4Exp 的原始支持 PR #53896（Qwen4Exp 派生自 Qwen3-Next，仍继承 `Qwen3NextAttention`/`Qwen3NextSparseMoeBlock`）；并指出 turboquant config 的 `_get_full_attention_layer_indices` 存在同类盲点，作为独立既存问题未在本次扩大范围。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
