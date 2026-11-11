# PR #58094: [Bugfix][Spec Decode] Propagate rope hf_overrides to MTP draft

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58094](https://github.com/vllm-project/vllm/pull/58094) |
| 作者 | gokay-ai |
| 合并日期 | 2026-11-09 |
| labels | bug, mrv2, ready, speculative-decoding |
| 热度 | 👍 2 · 💬 36 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
dict 型 `--hf-overrides`(YaRN/rope)构建 draft ModelConfig 时被丢弃,target 用扩展 RoPE 而 MTP draft 停在原生 rope,超原生长度后接受率归零;现转发 rope 相关键。

## 问题现象
正文(Fixes #58080):以 dict 形式传入 `--hf-overrides`(即 YaRN / rope 的 CLI 路径)时,target 模型跑了扩展 RoPE,而 MTP draft 仍用原生编码配置;一旦超过 draft 原生 `max_position_embeddings`,draft 的接受率降到 0%。影响面是「dict 型 hf-overrides + speculative decoding(MTP draft)」组合的部署;callable 型 overrides 不受影响。

## 根因
正文披露:构建 speculative draft 的 `ModelConfig` 时,dict 型 `--hf-overrides` 被整体丢弃——此前的 `compose_draft_hf_overrides` 只把 callable overrides 应用于 draft,dict overrides 被当作 target 专属的 key patch,不做转发。于是 YaRN/rope 相关的覆盖只改了 target,draft 配置保持原生 rope 设置。作者还说明:#37443 针对同一 bug(#37435),但与当前 `compose_draft_hf_overrides` 冲突且会丢掉 #45352 的 callable 组合能力,故非重复实现。

## 修复方案
改 `vllm/config/speculative.py`(及 `vllm/config/model.py` 辅助方法改 classmethod),让 rope 相关 dict 键到达 draft 配置:

- 新增 `_ROPE_HF_OVERRIDE_KEYS`(`rope_scaling`、`rope_parameters`、`original_max_position_embeddings`、`rope_theta`、`partial_rotary_factor`)与 `_NESTED_HF_CONFIG_KEYS`(`text_config`、`language_config`、`llm_config`、`thinker_config`);
- `_select_draft_hf_overrides` 从 dict overrides 中挑出 rope 键(含 VL wrapper 嵌套层下的);
- `_apply_dict_then_arch_override` 先应用 dict rope 补丁、再走 draft 架构重写,保证嵌套键在 arch mapping 提升 `text_config` 之前生效;flat draft config 上跳过嵌套 wrapper,避免制造裸 `text_config` dict;
- `max_position_embeddings` 刻意排除在外:shared drafts 已由 `_maybe_override_draft_max_position_embeddings`(#63909)处理,以保住 draft 的 max-model-len clamp。

```python
_ROPE_HF_OVERRIDE_KEYS = frozenset(
    {
        "rope_scaling",
        "rope_parameters",
        "original_max_position_embeddings",
        "rope_theta",
        "partial_rotary_factor",
    }
)
```

callable overrides 路径不变;新增 4 个 CPU 配置级测试(14 passed)。

## 影响与回归风险
- 影响窗口:正文未说明受影响版本/提交范围,不写。
- 回归面:按 diff,变化集中在 draft hf_overrides 的组装:`compose_draft_hf_overrides` 对 Mapping 型输入改为先挑 rope 键再组合;非 rope 的 dict 键(如 `num_hidden_layers`)依旧不转发,`max_position_embeddings` 依旧 target-only,callable 路径与 `functools.partial` 可序列化机制保持。空 overrides 与非 Mapping 输入回退原路径。涉及 YaRN/rope 类 CLI 覆盖 + MTP draft 的配置正确性,配置级改动无需多 GPU 即可回归。

## 关联
- [#58080](https://github.com/vllm-project/vllm/pull/58080)
- [#37443](https://github.com/vllm-project/vllm/pull/37443)
- [#37435](https://github.com/vllm-project/vllm/pull/37435)
- [#45352](https://github.com/vllm-project/vllm/pull/45352)
- [#63909](https://github.com/vllm-project/vllm/pull/63909)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
