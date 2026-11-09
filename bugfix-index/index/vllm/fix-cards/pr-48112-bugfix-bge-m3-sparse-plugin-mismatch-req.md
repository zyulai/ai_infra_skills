# PR #48112: [bugfix] bge-m3-sparse-plugin mismatch requests

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48112](https://github.com/vllm-project/vllm/pull/48112) |
| 作者 | staugust |
| 合并日期 | 2026-07-10 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`bge_m3_sparse_plugin` 用 `embed_request_queue.pop(0)` 按序匹配请求,坏请求残留在队列导致后续请求 pooling 参数错位、漏返回 sparse 输出;移除队列改用固定参数修复。

## 问题现象
用 `bge_m3_sparse_plugin` 服务 bge-m3 时,插件按请求更新 pooling_parameters,而请求是通过 `embed_request_queue.pop(0)` 取出的。两种错误场景:
1. 一个坏请求(例如输入超过模型最大上下文长度)带着 `embed_task=dense` 到达,render 直接返回错误,但这个请求没有任何机会从 `embed_request_queue` 中移除,残留在队列里;
2. 随后一个正常请求带着 `embed_task=dense&sparse` 到达,`merge_pooling_params` 会把那个坏请求当作 `raw_request` 弹出使用,结果这个正常请求只返回 dense embeddings,不返回期望的 sparse 部分。
影响面是使用该插件的 bge-m3 serving 请求输出内容错误(请求间串扰)。

## 根因
PR 正文披露:插件把请求对象放入队列、再按 FIFO 顺序 `pop(0)` 与 pooling 输出配对;一旦有请求在中途失败(如超长输入被 render 拒绝),它不会出队,队列头与实际请求错位,后续请求读到别人的 `embed_task`/`use_activation`/`dimensions`,产生请求不匹配(mismatch)。

## 修复方案
改 `tests/plugins/bge_m3_sparse_plugin/bge_m3_sparse_processor/sparse_embeddings_processor.py`:
1. 删除 `embed_request_queue` 及其在 `pre_process`(online/offline)中的入队逻辑;
2. `merge_pooling_params` 不再 `pop(0)` 并校验 `raw_embed_request`,改为固定设置 `params.task = "embed&token_classify"`、`params.use_activation = True`、`params.dimensions = self.embed_dimensions`,并删掉原有的 matryoshka dimensions 校验块;
3. `post_process` 中 `embed_dimensions` 不再读 `raw_request.dimensions`,固定用 `self.embed_dimensions`;
4. 清理不再使用的 import(`ModelConfig`、`EmbedRequestMixin`、`EMBED_TASKS`)。

```diff
-        raw_embed_request = self.embed_request_queue.pop(0)
-        ...
-        params.use_activation = raw_embed_request.use_activation
-        if params.use_activation is None:
-            params.use_activation = True
-        params.dimensions = raw_embed_request.dimensions
+        params.task = "embed&token_classify"
+        params.use_activation = True
+        params.dimensions = self.embed_dimensions
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅该示例插件(位于 tests/plugins/)。按请求自定义 `use_activation`/`dimensions` 的能力被移除(现在固定 True / 模型 embedding_size),matryoshka 维度校验一并删除;换取消除了请求间错位,坏请求不再影响后续请求。验证:`test_bge_m3_sparse_io_processor_plugins.py` 5 passed。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
