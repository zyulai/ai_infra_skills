# PR #20289: [doc] fix the incorrect logo in dark mode

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20289](https://github.com/vllm-project/vllm/pull/20289) |
| 作者 | reidliu41 |
| 合并日期 | 2025-07-01 |
| labels | documentation, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
文档深色模式下误用浅色 logo;改为同时嵌入明暗两版图片并用 CSS 按配色方案切换显示。

## 问题现象
文档站首页(Welcome to vLLM)在深色模式下显示的仍是浅色版 logo 图片,与配色方案不匹配。触发场景:用户使用 mkdocs 的 dark(即 "slate")配色方案查看文档。正文未描述更详细的复现步骤或影响面。

## 根因
正文说明:按作者理解,深色模式应当使用 dark logo png,浅色模式使用 light logo(原文 "it should use dark logo png for dark mode (light with light logo)"),即此前 `docs/README.md` 里只硬编码引用了 `vllm-logo-text-light.png` 一张图,没有随配色方案切换的逻辑。

## 修复方案
- `docs/README.md`:在 `<figure>` 中同时放入两张 logo 图片,分别加 class `logo-light` 与 `logo-dark`。
- `docs/mkdocs/stylesheets/extra.css`:新增规则,浅色方案(default)下隐藏 `logo-dark`,深色方案(slate)下隐藏 `logo-light`:

```css
[data-md-color-scheme="default"] .logo-dark {
  display: none;
}

[data-md-color-scheme="slate"] .logo-light {
  display: none;
}
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:改动仅涉及 `docs/README.md` 与 `docs/mkdocs/stylesheets/extra.css`,影响文档站首页 logo 的渲染;不触及任何运行时代码路径。若页面元素未带对应 class 或配色 scheme 属性名变化,则切换规则不会命中(由 diff 推断)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
