# PR #62392: [Perf][AuxOutput] Do not rescan referenced keys in the routed-experts store

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aoshen02 |
| 合并 | 2026-10-19 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 2 个文件(store.py 与新增测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/62392 |

## 一句话结论

修复 routed-experts 输出存储驱逐时反复重扫 referenced key 的问题,RL rollout 场景吞吐 +17.7%,writer 线程 `put` 耗时从 526 s 降到 4.4 s。

## 动机

`BlockObjectStore` 把 referenced 与 unreferenced 对象放在同一个 LRU 里,驱逐时从最旧端开始扫描并跳过 referenced key。运行中请求引用的 key 会堆积在 LRU 前端,于是每次需要驱逐的 `put()` 都要在 writer 线程上重扫全部这些 key:#62044 实测 300k keys 时每次 `put` 约 9 ms,移除该重扫后 GLM-4.5-Air 20 个 RL step 的 writer 线程时间从 281 s 降到 2.3 s,吞吐 +15.6%。本 PR 从 #62044 拆出,只做 store 驱逐修复。

## 优化点

让 store 仿照 `BlockPool` 的 free block queue 组织:LRU 只持有 unreferenced 对象。具体机制:新增 `_slots` dict 保存每个 key 到 slot 的映射(此前 slot 直接存在 `_lru` 的 value 里);key 的引用计数从 0 变 1 时移出 LRU(`_retain` 中 `pop`),计数降回 0 时插回 LRU 尾部(`_release` 中重新插入);驱逐只从 LRU 前端取 key,天然跳过 referenced 对象,不再需要逐个检查 `_references`。同一 `put()` 内释放的 key 仍按 release 顺序排在批次对象之后,`get_concatenated` 也只对仍在 LRU 中的 key 做 `move_to_end`。语义上:retain/release 平衡时驱逐顺序、victim 与报错与原来完全一致;释放一个未被引用的 key 仍抛 `KeyError`,但同调用内早先释放的 key 会落到 LRU 尾部,后续 victim 选择可能不同。

## 关键改动

```text
vllm/distributed/aux_output_connector/store.py: _slots 与 _lru 分离,LRU 只含 unreferenced key,驱逐免重扫
tests/distributed/aux_output_connector/test_store.py: 新增不触碰内部结构的行为级回归测试
```

`put` 与 `_retain` 的核心变化(节选):

```python
self._slots.setdefault(key, self._UNALLOCATED_SLOT)
if key not in self._references:
    self._lru[key] = None
    self._lru.move_to_end(key)
...
def _retain(self, keys):
    for key in keys:
        references = self._references.get(key, 0)
        if not references:
            self._lru.pop(key, None)
        self._references[key] = references + 1
```

## 性能声明

PR 正文 benchmark(GLM-4.5-Air,单 GB200 节点 TP4,`gpu_memory_utilization=0.9`(2,434,064 KV tokens、167,341 store objects)、`max_model_len=20480`、routed experts 开启;每 step 为 64 prompts x 8 samples 的 RL 式 rollout,1k-16k 输出 token 且 `ignore_eos`,两侧同用 2026-10-16 nightly 镜像、仅 store.py 不同):吞吐 main 平均 9,162 tok/s 对 PR 10,788 tok/s,+17.7%(单表内 main 9,446 vs PR 10,788 tok/s);writer `put` 总时间 main 526.0 s / 617.7 s 对 PR 4.4 s / 7.3 s;单 step 复现脚本 459.3 s(main)对 417.7 s(PR),token 数相同。另引 #62044 的测量:+15.6% 吞吐、300k keys 时每 `put` 约 9 ms、20 step writer 281 s 对 2.3 s。小 store 时收益消失:`gpu_memory_utilization=0.5`(约 52k objects)下 writer 仅占每 step 17%,吞吐基本不变(5,868 vs 5,897 tok/s)。

## 验证与回归风险

`pytest tests/distributed/aux_output_connector/test_store.py` 65 passed。新增 `test_store_eviction_does_not_rescan_referenced_keys` 用统计自身 hash 次数的 `str` 子类 key:本改动下 referenced key 的 hash 计数为 0,`main` 上为 180,000(上限 900),且不触碰 store 内部字段(改 `_references`/`_lru` 名会使其失败)。作者另跑了 5,000 条随机 `put`/`get`/retain/release 序列的 differential fuzz(非 PR 的一部分):平衡引用下存储状态与驱逐顺序与 `main` 完全一致;含 unreferenced release 时 404 条序列在 `KeyError` 后选择不同 victim。回归风险:依赖释放顺序语义的场景 victim 可能变化,且 writer 线程在 `KeyError` 后停止——与原行为一致的错误路径,但 victim 差异需下游知晓。

## 关联

从 #62044 拆出(该 PR 含完整修复与 full-KV crash 修复);测试所在目录同属 aux_output_connector(routed experts 输出通道)。其他同分类 PR 未知。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
