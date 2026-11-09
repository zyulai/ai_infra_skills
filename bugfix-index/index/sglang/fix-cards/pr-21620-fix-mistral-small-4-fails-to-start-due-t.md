# PR #21620: fix: Mistral Small 4 fails to start due to config/weight format mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21620](https://github.com/sgl-project/sglang/pull/21620) |
| 作者 | LiYomi |
| 合并日期 | 2026-03-30 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 9 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Mistral Small 4 同时带 params.json 与 config.json,config 走 Mistral native 解析而权重却按 HF 名加载,重映射匹配不上导致 w_kc 为 None 崩溃;修复为该类模型名强制按 native 格式加载。

## 问题现象
`mistralai/Mistral-Small-4-119B-2603` 启动失败,报 `AttributeError`,因为 `w_kc` 是 `None`(正文 fixes #21611)。触发条件是该模型目录同时存在 `params.json` 和 `config.json` 两个配置文件,造成 config 解析路径与权重加载路径不一致,所有权重被当 Unrecognized 跳过。

## 根因
正文披露的根因链:1) 配置路径——`hf_transformers_utils.py` 匹配到 `"mistral-small-4"` → 调 `_load_mistral_large_3_for_causal_LM()` → 解析 `params.json` → 期望 Mistral native 权重名(如 `layers.X.attention.wkv_b.weight`);2) 权重路径——`_is_mistral_native_format()` 看到两个文件都存在 → 返回 `False` → `load_format` 保持 `"auto"` → 按 HF 格式权重名加载(如 `language_model.model.layers.X.self_attn.kv_b_proj.weight`);3) 结果——重映射正则匹配不上 HF 名,所有权重被跳过标记为 "Unrecognized" → `kv_b_proj` 从未加载 → `w_kc` 为 `None` → 崩溃。

## 修复方案
修改 `python/sglang/srt/server_args.py` 的 `_is_mistral_native_format()`:当 `params.json` 与 `config.json` 同时存在时,若模型名匹配会触发 `_load_mistral_large_3_for_causal_LM` 的模式(`mistral-large-3`、`mistral-small-4`、`leanstral`,与 `hf_transformers_utils.py::get_config/get_tokenizer` 中的名字检查保持同步),则返回 `True`,使 `load_format` 被设为 `"mistral"`(native 格式)。关键 diff:

```python
+        _MISTRAL_NATIVE_CONFIG_PATTERNS = (
+            "mistral-large-3",
+            "mistral-small-4",
+            "leanstral",
+        )
+        def _check_format(has_params: bool, has_hf_config: bool) -> bool:
+            if has_params and not has_hf_config:
+                return True
+            if has_params and has_hf_config:
+                model_lower = str(self.model_path).lower()
+                if any(name in model_lower for name in _MISTRAL_NATIVE_CONFIG_PATTERNS):
+                    return True
+            return False
```

本地目录与 HF hub 两条检测路径都改走 `_check_format`。另新增 `test/registered/models/test_ministral4_models.py`(GSM8K 阈值 0.9、MMMU 0.45,TP2)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 `_is_mistral_native_format` 的判定:只含 `params.json` 的模型(如 Mistral Large 3)原逻辑不变;两文件都在但名字不在模式列表的模型(如 Mistral-7B-v0.3)仍默认 HF 格式;只有名字命中三个模式的模型才改为 native 格式加载。正文确认单文件逻辑改动,不影响其他模型。

## 关联
- [#21611](https://github.com/sgl-project/sglang/issues/21611)(Mistral Small 4 启动崩溃 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
