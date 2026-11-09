# PR #17191: [BUGFIX] fix radix cache memory consumption to avoid OOM

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17191](https://github.com/sgl-project/sglang/pull/17191) |
| 作者 | yizhang2077 |
| 合并日期 | 2026-01-17 |
| labels | high priority, run-ci |
| 热度 | 👍 5 · 💬 7 |
| 分类 | KV Cache / Memory |
| bug 类型 | resource |

## 一句话结论
radix tree 切分时父子节点共享同一 value 视图,child 驱逐后显存无法回收,长前缀共享下动态 OOM;切片改为 clone 释放引用。

## 问题现象
radix tree 做 splitting 时,`new_node` 与 `child` 都持有插入 value 的视图(view),因此 child 被驱逐后,`child.value` 占用的显存要等路径上所有持有该 value 视图的节点都被驱逐才会归还 torch 内存池;树共享长前缀时可能导致动态 CUDA OOM。正文指出该 bug 也可能存在于其他 radix cache 类型。UT 度量:修复前 `torch_allocated_memory / tree total_size` 超过 400(异常),修复后小于 10(理论应接近 value dtype 的大小)。

## 根因
正文披露:splitting 时切片操作(`value[:split_len]`、`value[split_len:]`)产生的是同一底层张量的视图,父子节点都持有引用,驱逐单个节点无法释放底层显存,引用随切分不断滞留。

## 修复方案
在 `radix_cache.py`、`swa_radix_cache.py`、`mamba_radix_cache.py` 的 `_split_node` 以及 `radix_cache.py::_insert_helper` 中,对 value(及 host_value)切片后立即 `.clone()`,切分/插入后不再共享底层存储:

```python
-            new_node.value = child.value[:split_len]
-            child.value = child.value[split_len:]
+            new_node.value = child.value[:split_len].clone()
+            child.value = child.value[split_len:].clone()
```

另在 `test_radix_cache_unit.py` 新增 `test_memory_allocated`,构造 1 万条共享 10000 token 公共前缀的序列,断言 `torch_allocated < cache_size_bytes * 2`。

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:四类 radix cache(基础/SWA/HiRadix/Mamba)的节点切分与插入;代价是切分时的一次 clone 拷贝,作者认为开销小且可被 overlap schedule 掩盖;换来的显存占用与驱逐行为变化覆盖所有走这些 cache 的负载。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
