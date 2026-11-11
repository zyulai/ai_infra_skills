# PR #58094: [Bugfix][Spec Decode] Propagate rope hf_overrides to MTP draft

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gokay-ai |
| 合并 | 2026-11-09 |
| 分类 | Speculative Decoding(major) |
| Label | bug, speculative-decoding, ready, mrv2 |
| Diff | +155/-13,4 files(素材未提供 diffstat,按 pack diff 计数) |
| 链接 | https://github.com/vllm-project/vllm/pull/58094 |

## 一句话结论

修复 YaRN/RoPE 的 `--hf-overrides` dict 未传给 MTP draft 的 bug:超过 draft 原生 `max_position_embeddings` 后 acceptance 掉到 0% 的问题得以恢复。

## 动机

PR 之前,`--hf-overrides` 以 dict 形式传入(YaRN / rope CLI 路径)时,构建 speculative draft 的 `ModelConfig` 会把它整个丢弃:target 按扩展 RoPE 运行,draft 仍停留在原生编码,请求一旦超过 draft 原生 `max_position_embeddings`,接受率直接归零(Fixes #58080)。对用 YaRN 扩展上下文又开 MTP 投机解码的长上下文模型,这使投机解码在该区间完全失效。Callable 覆盖此前已转发给 draft,只有 dict 路径断了。

## 优化点

`SpeculativeConfig.compose_draft_hf_overrides` 重写 dict 分支:先用 `_select_draft_hf_overrides` 挑出 rope 相关 key(`rope_scaling`/`rope_parameters`/`original_max_position_embeddings`/`rope_theta`/`partial_rotary_factor`),并递归处理嵌套在 `text_config`/`language_config`/`llm_config`/`thinker_config` 等 VL wrapper 下的同名 key,深拷贝后转发给 draft。`max_position_embeddings` 刻意保持 target-only,避免破坏 #63909 的 draft max-model-len clamp;其余 dict patch 仍 target-only。新助手 `_apply_dict_then_arch_override` 先应用 dict patch、再做 draft 架构映射,保证嵌套 key 在 arch mapping 提升 `text_config` 之前生效;flat draft config 上跳过嵌套 wrapper,防止制造会被 `get_text_config()` 误取的裸 `text_config` dict。`ModelConfig._apply_dict_overrides`/`_update_nested` 相应改为 classmethod 以便跨类复用,组合结果仍保持可 pickle(draft `ModelConfig` 要发给 spawned engine-core 进程)。

## 关键改动

```text
vllm/config/speculative.py: 新增 _ROPE_HF_OVERRIDE_KEYS/_NESTED_HF_CONFIG_KEYS、_select_draft_hf_overrides 与 _apply_dict_then_arch_override,重写 compose_draft_hf_overrides 的 dict 分支
vllm/config/model.py: _update_nested/_apply_dict_overrides 改为 @classmethod
vllm/v1/worker/gpu/spec_decode/speculator_config.py: 同步 draft_hf_overrides 注释
tests/config/test_speculative_draft_hf_overrides.py: 新增 YaRN 转发/不转发用例
```

```python
_ROPE_HF_OVERRIDE_KEYS = frozenset({
    "rope_scaling", "rope_parameters",
    "original_max_position_embeddings",
    "rope_theta", "partial_rotary_factor",
})
```

## 性能声明

无 benchmark 数字。修复针对的现象是:target 带 YaRN dict 覆盖时,超过 draft 原生 `max_position_embeddings` 的请求 acceptance 为 0%;修复后 rope 覆盖到达 draft 配置(由 14 个 config 级单测断言,无量化收益数字)。

## 验证与回归风险

新增/修改 `tests/config/test_speculative_draft_hf_overrides.py`:14 passed;pre-commit(ruff、typos、mypy 3.10、SPDX、forbidden-imports)通过。纯 config 层,无需 multi-GPU/H20。风险面:`compose_draft_hf_overrides` 被所有 MTP/外部 drafter 的 draft `ModelConfig` 构建共用,转发规则变化影响一切带 dict hf_overrides 的投机解码;`max_position_embeddings` 不转发的 clamp 行为(#63909)与 callable composition(#45352)均有回归测试钉住;嵌套 key 在 flat draft config 上必须跳过。

## 关联

Fixes #58080。前序/相关:#63909(draft max-model-len clamp,本 PR 刻意不转发 `max_position_embeddings`);#45352(callable override 组合);#37443(同一 bug 的另一 PR,与现实现冲突且会丢 callable composition,未采用)。本批同分类:#62769。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
