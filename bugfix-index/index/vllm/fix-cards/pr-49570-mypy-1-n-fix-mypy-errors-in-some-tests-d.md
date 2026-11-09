# PR #49570: [MyPy][1/N] Fix mypy errors in some tests/ directories and enforce follow-imports=silent

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49570](https://github.com/vllm-project/vllm/pull/49570) |
| 作者 | hickeyma |
| 合并日期 | 2026-07-30 |
| labels | ready, v1, cpu, nvidia, mistral |
| 热度 | 👍 2 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
tests 的 mypy 用 `--follow-imports skip` 掩盖真实类型错误;修掉约 25 处后新增 SILENT_GROUPS 机制,让已修目录立即按 `follow-imports=silent` 强制检查。

## 问题现象
这是一组测试目录的静态类型检查问题,不是运行时 bug:MyPy 检查 `tests/*` 时用 `--follow-imports skip`,会隐藏真实的类型错误。修复后通过 `pre-commit run -a --hook-stage manual mypy-3.10`,mypy 通过。正文未描述运行时故障现象。

## 根因
正文披露两个机制问题:
1. 对 tests 目录用 `--follow-imports skip` 运行 mypy,掩盖了真实的类型错误。
2. `group_files()` 会把每个 `tests/**` 文件都归入 SEPARATE_GROUPS 中的 "tests" 条目,即使存在更具体的条目;因此单独把某个目录从 SEPARATE_GROUPS 移除并不能强制任何检查,必须等最后移除 "tests" 本身。

## 修复方案
1. 在 `tools/pre_commit/mypy.py` 新增 `SILENT_GROUPS` 列表,在 SEPARATE_GROUPS 之前匹配:已修复目录的文件进入默认组,立即按 pyproject 的 `follow-imports=silent` 检查,不必等最终清理。核心逻辑:

```python
silent_pattern = re.compile(f"^({'|'.join(SILENT_GROUPS)}).*")
...
if silent_pattern.match(changed_file):
    file_groups[""].append(changed_file)
    continue
```

2. SEPARATE_GROUPS 按最长前缀优先迭代(`sorted(SEPARATE_GROUPS, key=len, reverse=True)`),避免子目录组被父目录遮蔽。
3. 修复 21 个最便宜的 `tests/` 目录中约 25 个 mypy 错误,并把这 21 个目录移入 SILENT_GROUPS 强制执行。修法包括:`capability is not None` 空值检查、`MagicMock(spec=ModelConfig)` 替代 `SimpleNamespace`、用 `GDNAttentionMetadata` 构造替代 `SimpleNamespace`、`get_args(MTPModelTypes)`、`# type: ignore[arg-type]` 标注、`assert x is not None` 等;还把 `test_processor_error.py` 的触发输入从 `TokensPrompt([])` 改为 `ExplicitEncoderDecoderPrompt`(被 Processor 拒绝的编码器/解码器 prompt)。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:改动集中在 `tests/` 下约 25 个测试文件与 `tools/pre_commit/mypy.py`,不影响生产代码路径。风险点:测试文件的等价改写(如 MagicMock(spec=…)、metadata 构造参数补全)可能改变测试行为;`test_processor_error.py` 换了触发错误的方式;mypy 分组逻辑变更影响 pre-commit 时各 tests 目录的检查严格度。部分改动由 Bob 与 Claude Code AI 助手起草,经人工审查与测试验证。

## 关联
- Partial #49569(计划文档)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
