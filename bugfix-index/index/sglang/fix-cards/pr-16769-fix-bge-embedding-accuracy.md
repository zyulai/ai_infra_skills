# PR #16769: fix bge embedding accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16769](https://github.com/sgl-project/sglang/pull/16769) |
| 作者 | amote-i |
| 合并日期 | 2026-01-09 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

BGE 类 embedding 模型 pooling 取错;is_embedding 时改用 CLS pooling 而非 LAST。

## 问题现象

正文未描述具体现象(Motivation、Modifications、Accuracy Tests 等小节均为空模板,正文未给出评测数字)。从标题看:BGE embedding 模型的输出精度有问题。

## 根因

正文未披露根因。由 diff 可描述机制:`python/sglang/srt/models/bert.py` 中,当模型没有 `BertPooler`(`use_bert_pooler` 为假)时,`Pooler` 固定使用 `PoolingType.LAST` 对最后一个 token 做池化;而 BGE 类 embedding 模型以 `[CLS]` token 的表示作为句向量,应使用 CLS pooling,取错池化位置导致 embedding 精度异常(由 diff 推断)。

## 修复方案

改 `python/sglang/srt/models/bert.py`:在构造 `self.pooler` 前按服务器用途选择池化类型,embedding 服务用 CLS,其余保持 LAST:

```diff
+        pooling_type = (
+            PoolingType.CLS
+            if get_global_server_args().is_embedding
+            else PoolingType.LAST
+        )
         self.pooler = (
             BertPooler(config)
             if self.use_bert_pooler
-            else Pooler(pooling_type=PoolingType.LAST, normalize=True)
+            else Pooler(pooling_type=pooling_type, normalize=True)
         )
```

同时新增 `get_global_server_args` 的导入。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:走 `bert.py` 且未使用 `BertPooler` 的模型。行为变化:`--is-embedding`(embedding 服务)下此类模型改用 CLS pooling;非 embedding 场景(如分类/打分)维持 LAST,不受影响;`use_bert_pooler` 为真的模型路径不变。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
