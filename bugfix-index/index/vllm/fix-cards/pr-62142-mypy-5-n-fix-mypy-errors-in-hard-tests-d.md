# PR #62142: [MyPy][5/N] Fix mypy errors in hard tests/ dirs and enforce follow-imports=silent

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62142](https://github.com/vllm-project/vllm/pull/62142) |
| 作者 | adarshxs |
| 合并日期 | 2026-10-20 |
| labels | rocm, structured-output, speculative-decoding, ready, tool-calling, llama, qwen, deepseek, cpu, nvidia, dflash, minimax |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
8 个 tests/ 目录存在 533 个 mypy 错误且检查可回退,全部修清并改入 --follow-imports=silent 强制分组。

## 问题现象
启用的 mypy 覆盖(`tests/` 系列,#49569)中,`tests/entrypoints/llm`、`tests/entrypoints/openai`、`tests/kernels`、`tests/lora`、`tests/tool_parsers`、`tests/v1/attention`、`tests/v1/e2e`、`tests/v1/ec_connector` 这八个目录此前被放在 SEPARATE_GROUPS 单独检查,未强制 `--follow-imports silent`,共存在 533 个 mypy 错误(出现在 111 个文件;issue 早期估计约 650,后被 #59428 关闭 tests/ 下的 arg-type/assignment 后降为 533),无法防止回归。纯类型问题,不影响运行时行为。

## 根因
正文披露:这些目录此前只纳入 SEPARATE_GROUPS 分组,检查强度不足且不能防回退;目录内积累的 533 个错误包括大量 `X | None` 未判空、联合类型未收窄、fixture/辅助函数缺注解、mock 方法直接赋值等,另有 stricter 检查暴露出的既有测试缺陷(如 `StructuredOutputsParams(backend=...)` 不存在该字段会 TypeError、`tokenizer.get_vocab = MagicMock()` 永久污染模块级 fixture、mock prefill backend 签名与 ABC 不匹配等,正文 Details 一一列出)。

## 修复方案
- 将 8 个目录从 SEPARATE_GROUPS 移入 SILENT_GROUPS(`tools/pre_commit/mypy.py`,被吸收的 `tests/kernels/*` 子条目删除,只保留父条目,路由-only 改动);
- 按常规配方修复:`assert x is not None`、`assert isinstance(...)` 收窄、补注解,49 处 mock 方法赋值改为 `monkeypatch.setattr`;
- 新增 29 个带错误码与原因的 scoped `# type: ignore[...]`,删除 12 个不再抑制任何错误的 ignore;不用 `cast()`;
- 修复正文列出的既有测试缺陷(如 test_struct_output_generate 去掉非法 backend kwarg、test_metadata 对齐 `(transfer_id, block_ids, shape)` 三元契约、test_mla_noncausal 的 ClassVar 赋值移除等);
- 明确遗留:tests/kernels/moe/utils.py 中 `moe_quantize_weights_2d` 的 int8 non-block 分支调用了签名已变的 `ops.scaled_int8_quant(w, w_s, use_per_token_if_dynamic=...)`(现为 `(input, scale, azp, symmetric)`),若触达会 raise;无测试触达且需 GPU 验证,加注释与 scoped ignore 留 follow-up。
注:本卡 pack 的 diff 在 40000 字符处截断,以上修复项以正文 Details 为准,未展示部分未编造。

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:正文声明纯类型改动、不改 `vllm/`,不影响输出、精度与服务;影响面为测试代码与 CI 检查强度(未来这些目录的测试改动必须过 mypy),以及被修改测试文件的执行路径(monkeypatch 改写、assert 收窄);遗留的 moe_quantize_weights_2d 坏分支是已知隐患。

## 关联
- [#49569](https://github.com/vllm-project/vllm/pull/49569)(mypy on tests/ 系列)
- [#49570](https://github.com/vllm-project/vllm/pull/49570)(PR 0/1)、[#51043](https://github.com/vllm-project/vllm/pull/51043)(PR 2)
- [#59428](https://github.com/vllm-project/vllm/pull/59428)(关闭 tests/ 下 arg-type/assignment)
- [#26533](https://github.com/vllm-project/vllm/pull/26533)(Partial)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
