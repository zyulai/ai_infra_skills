# PR #56300: [Bugfix][Bench] Fix bench mm-processor crash in shared request sampling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56300](https://github.com/vllm-project/vllm/pull/56300) |
| 作者 | NickLucche |
| 合并日期 | 2026-09-11 |
| labels | bug, performance, qwen |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Sampler / Detokenization |
| bug 类型 | crash |

## 一句话结论
bench mm-processor 缺少 throughput 旧 flag 且默认 0 被 or 逻辑吞成 None,共享采样路径崩溃;改为可选读取并用身份判断保留显式 0。

## 问题现象
`vllm bench mm-processor --dataset-name random-mm` 在共享请求采样路径崩溃,报错:

```
File ".../vllm/benchmarks/datasets/datasets.py", line 705, in get_prefix
    if prefix_len <= 0:
TypeError: '<=' not supported between instances of 'NoneType' and 'int'
```

正文给出了完整复现命令(random-mm、num-prompts 200、random-input-len 32、bucket config {(224,224,1):1.0} 等),影响面是使用 mm-processor 子命令跑随机多模态数据集的基准测试。

## 根因
正文披露,`_to_serve_args`(vllm/benchmarks/throughput.py)有两个 bug:
1. 它直接读 `args.input_len`、`args.prefix_len`、`args.backend`、`args.no_oversample`,但 `mm_processor.add_cli_args` 从未定义这些 flag(遇到第一个就 AttributeError)。
2. `getattr(args, "random_prefix_len", None) or prefix_len` 把默认的 `--random-prefix-len 0` 当 falsy 强制成 None,None 被传给 `RandomMultiModalDataset.get_prefix` 触发上述 TypeError;`random_input_len`/`random_output_len` 存在同样的 truthiness-vs-identity 问题。

## 修复方案
改 `vllm/benchmarks/throughput.py` 的 `_to_serve_args`:
- 旧 flag(`input_len`/`output_len`/`prefix_len`)改用 `getattr(args, ..., None)` 可选读取,兼容未定义这些 flag 的 mm-processor CLI。
- `random_prefix_len`/`random_input_len`/`random_output_len` 回退逻辑改用身份判断(`getattr(args, "random_prefix_len", prefix_len)` 等),显式 0 不再被吞成 None;`hf_output_len`/`sharegpt_output_len`/`sonnet_*` 同步使用可选读取值。
- `backend` 缺省为 `vllm-chat`(mm-processor 走 `llm.chat`,必须用 chat 格式),并 `d.setdefault("no_oversample", False)`。

```diff
-    d["random_input_len"] = getattr(args, "random_input_len", None) or args.input_len
+    input_len = getattr(args, "input_len", None)
+    d["random_input_len"] = getattr(args, "random_input_len", input_len)
```

测试:tests/benchmarks/test_throughput_cli.py 新增 mm-processor 参数适配、旧 flag 回退、random-mm 采样不崩溃三个用例。正文注明本 PR 由 AI 辅助开发(Mistral Vibe),由提交者人工评审并端到端测试,#56277 为带此补丁的运行结果。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 benchmark 工具的 `_to_serve_args` 适配层;throughput 主路径(定义了旧 flag)行为不变(新回归用例覆盖旧 flag 回退);mm-processor 子命令从崩溃恢复为可用。

## 关联
- [#56277](https://github.com/vllm-project/vllm/pull/56277)(带补丁的运行结果)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
