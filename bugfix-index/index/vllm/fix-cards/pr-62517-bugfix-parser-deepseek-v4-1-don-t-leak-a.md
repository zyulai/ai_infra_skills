# PR #62517: [Bugfix][Parser] DeepSeek V4.1: don't leak a `<｜DSML｜ tool_calls>` opener into content

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62517](https://github.com/vllm-project/vllm/pull/62517) |
| 作者 | YaashuDave |
| 合并日期 | 2026-11-06 |
| labels | bug, tool-calling, deepseek, verified, DSv4.1 |
| 热度 | 👍 0 · 💬 19 |
| 分类 | Model-specific Fast Path |
| bug 类型 | resource |

## 一句话结论

DeepSeek V4.1 解析器只认 `<｜DSML｜ calls>` 开头,模型改用 V4 的 `<｜DSML｜ tool_calls>` 开头时打开符漏进 content;把它加为 TOOL_START 变体。

## 问题现象

DeepSeek V4.1 输出的工具块若以 `<｜DSML｜ tool_calls>`(V4 的包裹名 + V4.1 的空格写法)开头,该开头标记会原样出现在 `content` 里;流式与非流式均如此。作者用 #62051 的输入在 `DeepSeekV41EngineFolder`/解析器上验证(CPU,非流式与逐字符流式加 `finish_streaming()`):对 `\n\n<｜DSML｜ tool_calls>\n<｜DSML｜ invoke name="WebFetch">…</｜DSML｜ tool_calls>`,main @ `bde4d00113` 能解析出 call `WebFetch {"url": "https://x.com"}`,但 `content` = `"\n\n<｜DSML｜ tool_calls>\n"`;本 PR 后同样的 call 被解析且开头不再进 `content`。规范化写法 `<｜DSML｜ calls>`、无空格 V4 写法、截断调用三种输入行为不变(无空格 V4 块仍按文本返回)。

## 根因

正文明确披露:`deepseek_v41_config` 把 V4 的 `TOOL_START`(规范化写法加 `DSML_TOOL_START_VARIANTS`)替换成了单一字符串 `<｜DSML｜ calls>`。当块以 `<｜DSML｜ tool_calls>` 开头时,其后的 V4.1 invoke 仍能通过既有的 `(CONTENT, INVOKE_PREFIX)` 状态转移被匹配到,所以调用本身能解析;但前面的打开符停留在 `CONTENT` 状态,被当作普通文本输出,即"漏进 content"。

## 修复方案

改 `vllm/parser/deepseek_v41.py`:新增 `DSML_TOOL_START_VARIANTS: tuple[str, ...] = ("<｜DSML｜ tool_calls>",)`,并在 `deepseek_v41_config` 的 `terminals` 里把 `TOOL_START` 设为 `(DSML_TOOL_START, *DSML_TOOL_START_VARIANTS)`;注释说明 `terminal_overrides` 仍保留单一写法因为 `token_id_terminals` 复用它,变体只加在这一处。做法与 #56141 给 V4 配置添加拼错包裹名的方式一致。`token_id_terminals` 不变:V4.1 tokenizer 里只有 `｜DSML｜` 是新增 token(非 special),标签以文本形式到达解析器。

```diff
+DSML_TOOL_START_VARIANTS: tuple[str, ...] = ("<｜DSML｜ tool_calls>",)
 DSML_TOOL_END = "</｜DSML｜ calls>"
```

效果上,这个打开符也像规范的 `<｜DSML｜ calls>` 一样会结束一个打开的 reasoning 块。新增测试 `test_v4_wrapper_name_opens_tool_block`(流式 chunk 1/7/10000、非流式、thinking 开/关)与 `test_v4_wrapper_name_ends_reasoning`,7 个用例在 main 上全部失败、本 PR 全部通过;三个 DeepSeek 解析器测试文件 147 例全过,`tests/parser` 与 `tests/tool_parsers` 共 8064 passed(较 main 多 7 个新测试)。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;问题自 V4.1 解析器配置以单一 `TOOL_START` 字符串取代 V4 变体机制后存在。
- 回归面:按 diff,只改 `deepseek_v41` 解析器配置:散文中提到 `<｜DSML｜ tool_calls>` 现在会像提到 `<｜DSML｜ calls>` 一样打开工具块(作者注明这与 #56141 对 V4 变体的取舍一致)。Rust 前端只在识别出的块内解析 invoke,该块仍整体按文本返回,本 PR 不改变此行为;完全 V4 无空格拼写仍按文本返回(与 Rust 前端 `v4_tags_remain_text_in_v41_dialect` 一致)。与 #61142、#62259 修改同一配置文件,后落地者需小规模 rebase。作者仅在 CPU 上做了 parser 级测试,未实际 serve DeepSeek-V4.1-Flash。

## 关联

- https://github.com/vllm-project/vllm/issues/62051 (来源 issue,"V4 spelling" 行)
- https://github.com/vllm-project/vllm/pull/56141 (V4 变体的先例做法)
- https://github.com/vllm-project/vllm/pull/62259 (#62051 截断行,同文件)
- https://github.com/vllm-project/vllm/pull/61142 (另一处 V4.1 opener 漂移,同配置)
- https://github.com/vllm-project/vllm/pull/56208 (Rust 前端 V4 标签保持文本)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
