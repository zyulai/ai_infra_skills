# PR #21407: [FIX] Flux2-Klein prompt tokenization length to 512 and add regression coverage

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21407](https://github.com/sgl-project/sglang/pull/21407) |
| 作者 | adityavaid |
| 合并日期 | 2026-03-28 |
| labels | run-ci, diffusion |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
Flux2-Klein 误继承 Flux1 的 max_length=77 截断长 prompt,改为显式 512 并加防回归守卫与测试。

## 问题现象
Flux2-Klein 的 prompt 在 tokenization 阶段被按 `max_length=77` 截断。Flux2-Klein 使用的是 Qwen3/Mistral 风格的 text encoder 路径,77 这个长度对它是错误的,长 prompt 因而被截断,影响该模型文生图输入的完整性。问题由 issue #21372 报告,本 PR 将其关闭。

## 根因
正文披露:旧行为来自从 Flux1 继承的 `text_encoder_extra_args`——Flux1 是 CLIP 风格的 77 长度,Flux2-Klein 配置沿用了这一继承值,导致其 tokenization 以 `max_length=77` 运行。这是配置继承引入的回归。

## 修复方案
改 `python/sglang/multimodal_gen/configs/pipeline_configs/flux.py`:
- `Flux2PipelineConfig` 显式声明自己的 `text_encoder_extra_args`,使用 `max_length=512`、`padding="max_length"`、`truncation=True`,不再继承 Flux1 的 CLIP 风格 77;
- 加固 `Flux2KleinPipelineConfig.tokenize_prompt`:丢弃调用方传入的 `max_length`,强制固定为 512,防止合并进来的 kwargs 悄悄把 77 带回来(defensive guard);
- 新增 `test_flux2_klein_tokenization.py` 回归测试,防止未来回归。

关键摘录:
```diff
-        max_length = tok_kwargs.pop("max_length", 512)
+        tok_kwargs.pop("max_length", None)
+        # Flux2 Klein uses max_length 512.
+        max_length = 512
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动集中在 Flux2 家族 pipeline 配置与 Klein 的 `tokenize_prompt`。长于 77 token 的 prompt 从此不再被截断;调用方显式传入的 `max_length` 会被忽略(有意为之的守卫行为);Flux1 及其他家族配置不受影响。

## 关联
- closes https://github.com/sgl-project/sglang/issues/21372

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
