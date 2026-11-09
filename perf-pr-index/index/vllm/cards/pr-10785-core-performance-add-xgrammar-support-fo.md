# PR #10785: [Core][Performance] Add XGrammar support for guided decoding and set it as default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aarnphm |
| 合并 | 2024-12-03 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | documentation, performance, frontend, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/10785 |

## 一句话结论

V0 guided decoding 新增 XGrammar 后端并设为默认，A100 JSON grammar 下吞吐 0.94 对 outlines 0.22 req/s。

## 动机

PR 正文说明为 V0 增加 XGrammar 的初始支持，并把 grammar 与 json 用法的默认后端设为 XGrammar（与 @mgoin 合作完成）。正文未描述改用默认之前的具体瓶颈，但给出与 outlines 后端的对比数据。

## 优化点

新增 XGrammar guided decoding 实现 `vllm/model_executor/guided_decoding/xgrammar_decoding.py`，在 `requirements-common.txt` 加入 `xgrammar` 依赖、在 `docs/source/conf.py` 加入 `xgrammar`，并在 `vllm/model_executor/guided_decoding/__init__.py`、`vllm/config.py`、`vllm/engine/arg_utils.py`、`vllm/engine/llm_engine.py`、`vllm/engine/async_llm_engine.py`、`vllm/engine/multiprocessing/client.py` 相应接入 XGrammar 并设为默认。测试侧在 `test_guided_processors.py` 的 backend 参数化中加入 `xgrammar`，并新增 `test_guided_json_object` 断言输出可被 `json.loads` 解析为 dict。

## 关键改动

```text
vllm/model_executor/guided_decoding/xgrammar_decoding.py: 新增 XGrammar guided decoding 后端
vllm/model_executor/guided_decoding/__init__.py: 接入 XGrammar 并设为 grammar / json 默认后端
requirements-common.txt / docs/source/conf.py: 新增 xgrammar 依赖与文档配置
tests/entrypoints/llm/test_guided_generate.py / tests/model_executor/test_guided_processors.py: 新增 json_object 用例并扩展 backend 参数化
```

## 性能声明

正文使用 #10557 的 benchmark 脚本，条件为 A100 80GB、Llama 3.2 3B、chunked prefill 开启、JSON grammar（NOTE 原文）。XGrammar 后端：Throughput 0.94 requests/s、1022.46 total tokens/s、480.27 output tokens/s、Correct rate 100.0%；First token latency mean 4552.21 msecs（count 10、min 3289.77、50% 5102.69、max 5114.35）；Next token latency mean 11.91 msecs。对比 outlines：Throughput 0.22 requests/s、241.22 total tokens/s、113.31 output tokens/s、Correct rate 100.0%；First token latency mean 38533.08 msecs；Next token latency mean 12.96 msecs。pack 的 regex 未检测到定量性能声认（正文未用 speedup 等字样），以上为正文给出的两组后端对比原始数值；PR 未注明精度（dtype）与更细的 workload 条件。

## 验证与回归风险

PR 正文未注明测试计划或测试结果。diff 新增 `test_guided_json_object`（断言生成文本可被 `json.loads` 解析为 dict）并把 `test_guided_logits_processor_black_box` 的 backend 参数化扩展到 `xgrammar`、`outlines`、`lm-format-enforcer`。正文未说明回归风险；默认后端切换会影响 V0 下 grammar 与 json 的 guided decoding 路径，diff 中 `vllm/config.py` 的改动仅为去掉若干行尾空白，属格式清理；两组后端的 Correct rate 均为 100.0%。正文未逐一列出受影响入口，也未说明默认后端切换是否影响其他 guided decoding 后端（如 lm-format-enforcer）的可用性。

## 关联

正文明确提到 benchmark 脚本来自 #10557，以及本改动与 @mgoin 合作完成。除此之外无明确 PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
