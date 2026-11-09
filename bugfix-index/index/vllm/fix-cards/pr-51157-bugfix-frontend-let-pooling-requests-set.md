# PR #51157: [Bugfix][Frontend] Let pooling requests set padding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51157](https://github.com/vllm-project/vllm/pull/51157) |
| 作者 | Hert4 |
| 合并日期 | 2026-08-27 |
| labels | bug, documentation, frontend, ready, multi-modality, verified |
| 热度 | 👍 3 · 💬 29 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
Pooling 在线接口的字段白名单缺 `padding`,`/v1/embeddings` 无法像离线 `LLM.embed` 那样要求 pad,SigLIP 等固定长度训练模型得到错误但形状正确的向量;给 pooling 请求新增 `padding` 字段并经 `TokenizeParams.with_kwargs` 透传。

## 问题现象
Pooling endpoints 在 `_build_pooling_tok_params` 中按固定字段列表构造 `TokenizeParams`,该列表没有 `padding` 字段,因此调用方没有任何值能让 `/v1/embeddings` 对 prompt 做 padding;离线 `LLM.embed` 支持 `tokenization_kwargs={"padding": "max_length"}`,在线则无等价能力。对「以固定序列长度训练、无 attention mask」的模型,不 pad 的结果是静默错误:SigLIP 即暴露此问题的案例——padding token 属于输入的一部分,且 pooled embedding 取自最后一个位置,未 pad 的文本 embedding 与图像 embedding 不可比。请求仍返回 200、向量形状正确,只有数值错误。实测(serving `google/siglip2-base-patch16-224 --runner pooling --max-model-len 64`,与 HF 参考按 `padding="max_length", max_length=64` 比较):带 padding 的 cosine 为 1.000000,不带为 0.616077 / 0.644452,不可用于检索。

## 根因
PR 正文披露:`_build_pooling_tok_params` 用固定字段清单构建 `TokenizeParams` 而非透传 `tokenization_kwargs`,清单里没有 padding 字段,导致在线请求无从表达 padding 需求(离线支持的 `tokenization_kwargs` 在线没有对应入口)。

## 修复方案
1. `vllm/entrypoints/pooling/base/protocol.py`:`PoolingBasicRequestMixin` 新增 `padding: Literal["max_length", "do_not_pad"] | None` 字段(采用 Transformers 的命名);`_build_pooling_tok_params` 重构为按分支构造 `tok_params` 后,若 `self.padding` 非 None 则经既有的 `tok_params.with_kwargs(padding=self.padding)` 透传——这正是 vLLM 翻译 Transformers 词汇的地方,无需新增映射逻辑;
2. 两个 SigLIP 示例(`vision_embedding_offline.py` / `vision_embedding_online.py`)的 `run_siglip` 改为带 `padding: "max_length"` 做 text embedding(原示例不带 padding,恰好演示的是 bug 本身);
3. `tests/entrypoints/pooling/embed/test_online.py` 新增 `test_padding`:断言 `max_length` 提高 prompt token 数、`do_not_pad` 与默认一致、非法值(如 `"longest"`)被拒(BadRequestError)。

设计取舍(正文):采用封闭集合而非自由字典,客户端可发送的值保持显式;Transformers 的 `padding=False` 故意不接受——`with_kwargs` 的真值检查会把 `False` 弹出并静默忽略,拒绝非法值好过静默无效。

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:`_build_pooling_tok_params` 被重构(早退分支合并为 if/else 后统一处理 padding),影响所有 pooling 请求的 tokenize 参数构建;不传 `padding` 时(默认 None)行为与之前完全一致。新增字段是封闭枚举,非法值返回 400。示例文件仅改变演示输出方式。测试结果:GB10(sm_121, aarch64)上 `test_padding` 与 `test_truncate_prompt_tokens` 均通过(2 passed in 33.18s)。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
