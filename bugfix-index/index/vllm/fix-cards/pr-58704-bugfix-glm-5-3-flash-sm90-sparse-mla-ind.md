# PR #58704: [Bugfix][GLM-5.3-Flash] SM90 sparse MLA: index_kpool mismatch leads to corruption via unread query token

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58704](https://github.com/vllm-project/vllm/pull/58704) |
| 作者 | mmastrac |
| 合并日期 | 2026-09-25 |
| labels | bug, ready, nvidia, glm |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
SM90 sparse MLA builder 的 `index_kpool` 误取 KV cache spec 的 `tokens_per_state`(恒为 1),导致超过 2048 token 后尾部 KV pool 未被读取、query token 被 attention 跳过,产生输出损坏;改为从模型配置读取 `index_kpool`。

## 问题现象
GLM-5.3-Flash 在中等长度上下文出现 corruption(输出损坏)。具体表现为:在超过 2048 tokens 的位置上,四个位置中有三个的 query token 及其前面最多两个 token 未被 attention 读取。影响量化(重度 tool calling 场景):约 1/8 的 34k sample sessions 出现 tool call 失败,约 1/2 的 42k sample sessions 失败。作者注明该比例随 workload/tokens 等高度变化,修复后在上述长度及其他长度上损坏的 tool call 均有改善。

## 根因
PR 正文披露:这是从 #58454 开始排查的又一处损坏来源,属于 #56868 与 #56605 的部分修复。根因是 `index_kpool` 取值来源错误——metadata builder 从 KV cache spec 读取 kpool,而该 spec 的 `tokens_per_state` 为 1,导致尾部 pool(tail pool)从未被读取(新增测试的 docstring 亦如此描述),于是上下文超过 `index_topk`(2048)后,按 `ctx % kpool` 计算出的 KV 长度错失尾部 token,部分 query token 未被 attention 覆盖。

## 修复方案
修改 `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` 的 `FlashInferMLASparseSM90Builder.__init__` 与 `_kv_lens_host`:

```diff
-        self._index_kpool = int(kv_cache_spec.tokens_per_state)
+        self._index_kpool = int(getattr(hf_config, "index_kpool", 1) or 1)
```

```diff
-        kpool = max(self._index_kpool, 1)
-        lens = torch.where(ctx <= topk, ctx, topk + ctx % kpool)
+        lens = torch.where(ctx <= topk, ctx, topk + ctx % self._index_kpool)
```

即 `index_kpool` 改从模型配置 `hf_config.index_kpool` 读取,DeepSeek-V3.2 默认为 1;`or 1` 兜底配置缺失或为 0 的情况,替代原先的 `max(..., 1)` 保护。同时在 `tests/v1/attention/test_flashinfer_mla_sparse_sm90.py` 新增 `test_builder_kpool_from_model_config`,以 kpool=4 / None / 无属性三种参数化验证 `_kv_lens_host` 的行数与 lens 输出。

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:仅影响 FlashInfer SM90 sparse MLA 后端的 metadata 构建路径(`flashinfer_mla_sparse_sm90.py`)。对配置了 `index_kpool` 的模型(GLM-5.3-Flash 等),KV 长度计算改为按真实 kpool 取模,行为变化即本修复目标;对配置中没有 `index_kpool` 的模型(如 DeepSeek-V3.2)默认为 1,与旧路径(`tokens_per_state=1`)等价。`or 1` 兜底替代 `max()` 意味着 kpool 不再可能小于 1。测试文件改动不进入运行时路径。

## 关联
- #58454: https://github.com/vllm-project/vllm/pull/58454 (排查起点)
- #56868: https://github.com/vllm-project/vllm/issues/56868 (partial fix)
- #56605: https://github.com/vllm-project/vllm/issues/56605 (partial fix)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
