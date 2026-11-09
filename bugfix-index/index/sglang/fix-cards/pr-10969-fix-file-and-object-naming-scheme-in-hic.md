# PR #10969: fix file and object naming scheme in HiCacheNixl to avoid data corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10969](https://github.com/sgl-project/sglang/pull/10969) |
| 作者 | ziruiliu |
| 合并日期 | 2025-10-10 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
HiCacheNixl 不感知 tp_rank/tp_size,TP>1 时多 worker 互覆写同一文件/对象致缓存数据损坏,现按 HiCacheFile 命名方案加后缀隔离。

## 问题现象
HiCache NIXL 存储后端(`python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py`)在 tp_size > 1 时,不同 TP worker 反复写入同一个文件/对象;这些被覆写的数据在缓存加载并复用(cache data is loaded and reused)时造成数据损坏(data corruption)。

## 根因
正文披露:HiCache NIXL 存储后端没有使用 engine 传入的 storage_config——后端完全不知道 tp_rank 与 tp_size,因此写文件/对象的命名不含 TP 维度,TP>1 时多个 worker 命中同一路径互相覆写,加载复用时读到被别的 rank 覆写过的数据,产生损坏。

## 修复方案
1. `storage/backend_factory.py`:nixl 后端创建时传入 `storage_config`(原先 `backend_class()` 不带参)。
2. `hicache_nixl.py`:`HiCacheNixl.__init__` 增加 `storage_config: HiCacheStorageConfig` 参数,按 HiCacheFile 类的命名方案构造后缀(MLA 模型仅 `_{model_name}`,否则 `_{model_name}_{tp_rank}_{tp_size}`),并新增 `_get_suffixed_key`:

```diff
+        if is_mla_model:
+            self.config_suffix = f"_{model_name}"
+        else:
+            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"
...
+    def _get_suffixed_key(self, key: str) -> str:
+        return key + self.config_suffix
```

3. `batch_get`/`batch_set`/`exists` 全部改用带后缀的 key(文件路径与对象名一致加后缀),读写与存在性查询对齐。
4. `test_hicache_nixl_storage.py` 同步构造 `HiCacheStorageConfig(tp_rank=0, tp_size=2, ...)` 验证。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:NIXL HiCache 存储后端的读写/查询路径。文件与对象名从此带模型名+TP 后缀,旧版本写入的存量缓存条目(无后缀)将无法被新代码命中(等于作废);构造签名变化要求所有 `HiCacheNixl(...)` 调用方传 storage_config;非 MLA 模型不同 TP rank 的缓存条目被隔离开。正文未提供 accuracy/benchmark 数据。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
