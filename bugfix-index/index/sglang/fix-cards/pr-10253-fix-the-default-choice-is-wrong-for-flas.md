# PR #10253: Fix: the default choice is wrong for flashinfer mxfp4 moe precision

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10253](https://github.com/sgl-project/sglang/pull/10253) |
| 作者 | LauYeeYu |
| 合并日期 | 2025-09-10 |
| labels | (none) |
| 热度 | 👍 1 · 💬 1 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

`--flashinfer-mxfp4-moe-precision` 的 choices 列表写错(默认值 `default` 不在其中),改为 `["default", "bf16"]`。

## 问题现象

正文未描述具体现象,只说明:该参数此前的定义「在使用 `ServerArgs.add_cli_args` 函数添加参数时会出问题」(The previous version will cause problem if use the `ServerArgs.add_cli_args` function to add argument),并指出默认值应为 `default` 而不是 "mxfp4"。

## 根因

正文披露的根因:`--flashinfer-mxfp4-moe-precision` 的取值集合写错,正确默认值应为 `default` 而非 "mxfp4"。(由 diff 推断)改动前 `choices=["mxfp4", "bf16"]` 而 `default=ServerArgs.flashinfer_mxfp4_moe_precision`,即类级默认值 `default` 不在 choices 允许范围内,经由 `add_cli_args` 构建 CLI 参数时会因此出问题(如参数校验失败)。

## 修复方案

只改 `python/sglang/srt/server_args.py` 中该参数的 choices:

```diff
-            choices=["mxfp4", "bf16"],
+            choices=["default", "bf16"],
             default=ServerArgs.flashinfer_mxfp4_moe_precision,
```

即 choices 从 `["mxfp4", "bf16"]` 改为 `["default", "bf16"]`,默认值保持 `ServerArgs.flashinfer_mxfp4_moe_precision` 不变。

## 影响与回归风险

- 回归面:仅影响 `--flashinfer-mxfp4-moe-precision` 的 CLI 参数定义;显式传 `--flashinfer-mxfp4-moe-precision mxfp4` 的调用方式将不再被接受(choices 中已无该值),需改传 `default` 或 `bf16`。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
