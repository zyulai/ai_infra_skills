# PR #62027: [Perf][Parser] Stream Qwen3 tool argument values incrementally (12 KB call: 2.1 s to 11 ms)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @deadstrobe5 |
| 合并 | 2026-10-17 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | ready, tool-calling, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/62027 |

## 一句话结论

Qwen3 工具参数改增量流式：12 KB 调用解析 2.1 s→11 ms、25 KB 8.4 s→22 ms，每 delta 约 4 µs。

## 动机

Streaming 一个 Qwen3 XML tool call 时，每个 delta 都会重解析整个参数：先走 `_qwen3_arg_converter`，再 `_safe_arg_prefix`，对整段 JSON 做逐字符 Python 循环。长参数值（例如 coding agent 写出的文件内容）随长度增长产生二次（quadratic）CPU 开销，落在 API server 上。本 PR 把 #61253（GLM）与 #61494（Gemma4）引入的 `arg_value_extender` hook 接到 Qwen3 grammar 上，消除这种随参数长度增长的重复解析。

## 优化点

extender 用与 converter 相同的 regex 读取正在展开的 open value，逐 delta 直接产出增量而不重跑 converter。当 open key 与前面某个 key 重复时拒绝启用（该 key 保持其在 JSON 中的首个位置）；当某个 delta 完成 `</parameter>` 或下一个 `<parameter=`（含空白变体、甚至跨 delta 拆分）时返回 None，交回 converter 接管。覆盖 `qwen3_config` 上的所有 parser（qwen3_coder、qwen3_xml、Seed-OSS、Step3p5、Nemotron-v3、Granite thinking、MiniCPM-V）；MiMo 使用不做 newline trimming 的变体，因其 converter 原样保留值。正文强调 streamed deltas 不变，包括 newline trimming、schema coercion 与 truncated calls。

## 关键改动

```text
vllm/parser/qwen3.py: 实现 _qwen3_arg_value_extender 并挂到 qwen3_config，读取 open value、处理 split closer 与重复 key
vllm/parser/mimo.py: MiMo 采用不去除换行的 extender 变体，保持值原样
tests/parser/engine/test_qwen3.py: 新增 TestArgValueExtender，覆盖长调用、转义、空行、闭合标签前缀与 split-closer 等用例
```
```text
    def test_extender_follows_newline_trimming_and_split_end(self):
        from vllm.parser.qwen3 import _qwen3_arg_value_extender

        extend = _qwen3_arg_value_extender("<parameter=k>")
        assert extend is not None
        deltas = ["\n", 'a"\n', "b", "\n", "</param", "eter>"]
        expected = ["", 'a\\"', "\\nb", "", "\\n</param", None]
        assert [extend(d) for d in deltas] == expected
```

## 性能声明

Benchmark 为 `Qwen3Parser.parse_delta`，每 delta 一个真实 Qwen3 token，Apple M5、Python 3.12、7 次交替运行：3 KB main 187-192 µs → 3.9-4.1 µs；12 KB 1.09-1.44 ms → 3.8-4.6 µs；25 KB 2.21 ms → 3.8 µs。Claims 给出：一个 12 KB `write_file` 调用（2,788 Qwen3 tokens）parser time 2.1 s → 11 ms，25 KB 调用 8.4 s → 22 ms；短三参数调用也约 10% faster（338 µs → 304 µs），因每个 delta 跳过一次转换。测试条件：Apple M5 / Python 3.12 / `Qwen3Parser.parse_delta` / 单 token per delta / 7 alternating runs；PR 未注明模型、精度与 GPU workload 条件（作者说明未能跑 GPU server 端到端）。

## 验证与回归风险

`pytest tests/parser/engine` 6026 passed；另通过 qwen3coder、step3p5、minicpm5xml 的 tool parser 测试，qwen3、nemotron_v3、granite_thinking、step3p5 的 reasoning 测试，以及 `test_streaming.py`、`test_parse.py`、`test_abstract_parser.py`；ruff 与 mypy 在改动文件通过。新测试 `TestArgValueExtender` 验证长 Qwen3/MiMo 调用与 `extract_tool_calls` 结果一致、converter 调用次数在 20→200 行保持恒定（在 main 上会增长）、closer 跨 delta 拆分、无 extender 的情形。本地差分 fuzz 流式 27,500 个随机 Qwen3/MiMo 调用、约 194K deltas，每个 `DeltaMessage` 完全一致；破坏 newline hold-back、split-closer window 或 leading-newline strip 会使 fuzz 失败。回归风险：作者未跑 GPU server 端到端，称模型输出 by construction 不变；未来改 `_PARAM_RE` 需同步改 `_PARAM_VALUE_END_RE`。

## 关联

正文明确采用 #61253（GLM）与 #61494（Gemma4）的 `arg_value_extender` hook。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
