# PR #50060: [Bugfix] Only pad transformers backend `value` when it is narrower

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50060](https://github.com/vllm-project/vllm/pull/50060) |
| 作者 | njhill |
| 合并日期 | 2026-07-28 |
| labels | bug, ready |
| 热度 | 👍 6 · 💬 0 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
两个 PR 叠加致 transformers backend 对 packed QKV 的 value 负 pad 被 F.pad 截断、key/value 宽度不匹配报错;改为仅在 value 更窄时才 pad。

## 问题现象
transformers backend 使用 packed grouped/multi-query QKV 投影(kv_heads > 1)的模型在注意力前向时报错(照抄):

```text
RuntimeError: The size of tensor a (16) must match the size of
              tensor b (8) at non-singleton dimension 1
```

CI 表现:`test_detects_and_rewrites_packed_qkv[2]` 在 #49987 自己的 CI(build 80569)通过,但从 build 80627 起每次构建都失败。正文解释了 CI 未能拦截的原因:#49987 比 #49982 合并早约 30 分钟测试,而触发该分支的 packed QKV 模块在 #49982 测试时尚不存在——两个 PR 各自单独都没错,只有都进入 main 后才互相作用。

## 根因
正文披露:#49982 在 `vllm_attention_forward` 中为 expanded MLA 添加了把 `value` pad 到 query/key head size 的逻辑,守卫条件是 `head_dim_v != head_dim_qk`;#49987 随后加入 packed grouped/multi-query QKV 支持,其注意力模块的 `key`/`value` 保持打包、不按 head 拆分,`value.shape[-1]` 为 `kv_heads * head_dim`,当 `kv_heads > 1` 时反而大于 query head size。此时 pad 量为负,`F.pad` 会静默截断而非 pad:`value` 被收窄到 query head size,而 `key` 保留全宽,注意力收到宽度不匹配的 key/value。

## 修复方案
改动 `vllm/model_executor/models/transformers/__init__.py` 的 `vllm_attention_forward`:守卫从「不相等就 pad」改为「仅在 value 真正更窄时才 pad」。更宽的最后一维只说明 `value` 未按 head 拆分、无需 padding,跳过后即恢复 #49982 之前的行为,expanded MLA 路径不变。

```python
# Pad `value` up to the query/key head size when it is smaller (expanded
# MLA). A larger last dim just means `value` isn't split per head, e.g.
# packed grouped/multi-query projections, and needs no padding.
pad_value = head_dim_v < head_dim_qk
if pad_value:
    value = F.pad(value.view(-1, head_dim_v), (0, head_dim_qk - head_dim_v))
    value = value.reshape(hidden, -1)
attn_output = self_attn.forward(query, key, value)
if pad_value:
    attn_output = attn_output.view(-1, head_dim_qk)[..., :head_dim_v]
```

## 影响与回归风险
- 影响窗口:#49982 与 #49987 均在 main 之后(自 CI build 80627 起失败)至本修复合入前。
- 回归面:仅 transformers backend `vllm_attention_forward` 的 value pad 分支;expanded MLA(`head_dim_v < head_dim_qk`)路径行为不变,packed QKV(`kv_heads > 1`,value 更宽)不再被截断。输出侧 unpad 也复用同一 `pad_value` 守卫,分支两侧逻辑保持一致。

## 关联
- [#49982](https://github.com/vllm-project/vllm/pull/49982)(引入 value padding)
- [#49987](https://github.com/vllm-project/vllm/pull/49987)(引入 packed QKV 支持)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
