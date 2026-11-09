# PR #22923: docs: fix incorrect default max-payload-size in gateway config reference

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22923](https://github.com/sgl-project/sglang/pull/22923) |
| 作者 | Wen-xuan-Xu |
| 合并日期 | 2026-04-16 |
| labels | documentation, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
文档把 `--max-payload-size` 默认值写成 256MB,实际代码为 512MB;修正 Configuration Reference 表格。

## 问题现象
sgl-model-gateway 的 Configuration Reference 文档表中,`--max-payload-size` 的默认值被写成 256 MB,而 Python 绑定(`router_args.py`)与 Rust 配置(`config/types.rs`)中的实际默认值均为 **512 MB**。正文说明该文档错误给排查 payload 大小问题的用户造成了困扰(见 #6622、#10800)。

## 根因
正文披露:文档表格记录的默认值(256MB)与代码实际默认值(512MB)不一致——文档写错,代码本身无需改动。

## 修复方案
仅修改 `docs/advanced_features/sgl_model_gateway.md` 的 Configuration Reference 表(纯文档改动,未动任何 Python 文件):

```diff
-| `--max-payload-size` | int | 256MB | Maximum request payload |
+| `--max-payload-size` | int | 512MB | Maximum request payload |
```

## 影响与回归风险
- 影响窗口:不适用(仅文档修正,不涉及运行时代码)。
- 回归面:仅影响 docs 页面 `docs/advanced_features/sgl_model_gateway.md` 中该参数行的展示;`router_args.py` 与 Rust `config/types.rs` 的实际默认值不变,运行行为零影响。

## 关联
- [#6622](https://github.com/sgl-project/sglang/issues/6622)(正文提到)
- [#10800](https://github.com/sgl-project/sglang/issues/10800)(正文提到)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
