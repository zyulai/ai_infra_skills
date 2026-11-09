# PR #25204: Fix frozen kv MTP crash when bonus_tokens is None

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25204](https://github.com/sgl-project/sglang/pull/25204) |
| 作者 | pyc96 |
| 合并日期 | 2026-05-14 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
Frozen KV MTP 不设置 bonus_tokens 导致取值崩溃,改从 input_ids 取最后已验证 token。

## 问题现象
启用 frozen KV MTP 投机解码时 server 直接崩溃(bonus_tokens 为 None)。正文给出的复现命令为:

```
python -m sglang.launch_server --model-path google/Gemma-4-26B-A4B-IT  --tp-size 1 --enable-torch-compile  --speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-26B-A4B-it-assistant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

即 Gemma-4-26B-A4B 模型 + NEXTN 投机算法 + torch compile 的场景。修复后 server 不再崩溃且 accept rate 正常。对应 issue [#24912](https://github.com/sgl-project/sglang/issues/24912)。

## 根因
正文披露:Frozen KV MTP 路径不会设置 `bonus_tokens`,而该字段是「最近投机重构 PR」引入/改动的,`select_last_verified_seed` 在选取最后已验证 token 时直接索引 `draft_input.bonus_tokens[last_indices]`,在 bonus_tokens 为 None 时即崩溃。也就是说崩溃源于投机解码重构后代码假设 bonus_tokens 总是存在,与 frozen KV MTP 的实际行为不符。

## 修复方案
改动 `python/sglang/srt/speculative/frozen_kv_mtp_utils.py` 的 `select_last_verified_seed`,把取最后已验证 token 的来源从 `bonus_tokens` 换成 `input_ids`:

```python
return (
    draft_input.input_ids[last_indices],
    draft_input.hidden_states[last_indices],
)
```

依据是正文的说明:frozen KV MTP 虽不设置 bonus_tokens,但可以从 input_ids 中拿到最后已验证的 token。

## 影响与回归风险
- 影响窗口:正文称 bug「Seems to be introduced recently in spec refactoring PRs」,即近期投机解码重构之后、本修复合并之前。
- 回归面:仅改动 frozen KV MTP 工具函数 `select_last_verified_seed` 中最后已验证 token 的取值来源,影响 frozen KV MTP(speculative NEXTN)路径下 seed 的选取;其他投机算法路径不受影响。若某路径下 input_ids 语义与 bonus_tokens 不一致,可能影响选取结果,但正文测试显示 accept rate 正常。

## 关联
- [#24912](https://github.com/sgl-project/sglang/issues/24912)(被修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
