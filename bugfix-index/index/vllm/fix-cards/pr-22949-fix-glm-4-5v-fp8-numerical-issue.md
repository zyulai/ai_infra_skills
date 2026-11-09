# PR #22949: Fix GLM-4.5V-FP8 numerical issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22949](https://github.com/vllm-project/vllm/pull/22949) |
| 作者 | zixi-qi |
| 合并日期 | 2025-08-19 |
| labels | documentation, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

GLM-4.5V-FP8 视觉 qkv 层量化豁免名单因层名前缀不匹配而失效,输出乱码;运行时前缀改为与量化配置对齐。

## 问题现象

用 vLLM 运行 GLM-4.5V-FP8 出现数值回归:视觉输入下模型输出完全乱码。作者用 `python examples/offline_inference/glm4_5v.py --num_prompts 3 --model glm45-fp8 --mm` 复现,修复前生成文本为 `', '`、`', , the., the '`、`',  connected by, '` 之类无意义内容;修复后能正确描述图片内容(如 "A duck swimming in water with its reflection visible")。作者另指出一个遗留问题:FP8 纯文本输出的质量仍低于 BF16 版本(出现复读循环),但"不像视觉那样是乱码"。

## 根因

正文披露:GLM-4.5V-FP8 的模型 config.json(HuggingFace zai-org/GLM-4.5V-FP8 仓库)想通过把 `visual.blocks.1.attn.qkv_proj` 加入量化豁免名单来关闭视觉注意力 qkv 投影层的量化;但 vLLM 运行时该层的实际前缀叫 `visual.blocks.1.attn.qkv`(没有 `_proj` 后缀)。两者不匹配导致豁免规则落空,这些层被错误量化,产生数值回归。

## 修复方案

作者称为"短期修复(short term fix)":改 `vllm/model_executor/models/glm4_1v.py` 中视觉注意力模块的 qkv 层前缀,使其与 GLM-4.5V-FP8 量化配置对齐——仅在存在 quant_config 时改用 `_proj` 后缀,保持非量化路径原名:

```diff
-            prefix=f"{prefix}.qkv",
+            # Change qkv prefix to align with GLM-4.5V-FP8 quantization config
+            prefix=f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv",
```

此外 diff 还在 `examples/offline_inference/vision_language.py` 与 `vision_language_multi_image.py` 中新增 `run_glm4_5v` / `run_glm4_5v_fp8`、`load_glm4_5v` / `load_glm4_5v_fp8` 示例并注册到 model_example_map,以及若干文档字符串尾随空格清理(与修复逻辑无关)。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;问题针对 GLM-4.5V-FP8 权重发布后的 vLLM 视觉路径。
- 回归面:按 diff,行为分支以 `quant_config` 是否存在划分——非量化模型前缀仍为 `.qkv`,路径不变;任何带量化配置的 GLM-4.1V/GLM-4.5V 系权重,若其 checkpoint 中该层权重名仍写作 `.qkv` 而非 `.qkv_proj`,加载时将因前缀不匹配而无法对应(作者称此改动正是为了对齐 FP8 官方量化配置,其他量化变体需确认权重命名)。纯文本推理不经过视觉注意力,不受影响。

## 关联

- https://huggingface.co/zai-org/GLM-4.5V-FP8/blob/main/config.json (正文引用的量化豁免配置)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
