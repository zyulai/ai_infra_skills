# PR #12001: perf: optimize TypeBasedDispatcher using dict for O(1) lookup

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zhengxle |
| 合并 | 2025-11-15 |
| 分类 | Other(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 utils.py 与 2 个新测试文件、run_suite 注册 |
| 链接 | https://github.com/sgl-project/sglang/pull/12001 |

## 一句话结论

`TypeBasedDispatcher` 的类型查找从 list 线性扫描改为 dict O(1),混合 workload 基准快 7.97x。

## 动机

`TypeBasedDispatcher` 用 list 线性扫描(O(n))为每个请求类型找 handler;代码库中已注册 30+ 类型,在高吞吐场景 dispatcher 被频繁调用,线性扫描成为瓶颈。本 PR 把底层数据结构从 list 换成 dict,类型查找从 O(n) 降到 O(1),并保持全部公共 API 与行为兼容。

## 优化点

`_mapping` 从 `List[Tuple[Type, Callable]]` 改为 `OrderedDict(mapping)`(保持注册顺序),`__call__` 先按 `type(obj)` 做精确 O(1) dict 命中;未命中再查 MRO 缓存;仍未命中才按注册顺序做 isinstance 遍历并写入 `_mro_cache`(含负缓存),保住对子类的 isinstance 语义——这是 review 中补上的关键修复,否则继承场景会派发失败。`__iadd__` 合并另一个 dispatcher 时按 dict 语义去重并清空 MRO 缓存。fallback 函数与异常行为保持不变;线程安全特性与原实现一致。内存上 dict 比 list 略大(绝对量 KB 级),正文称可忽略。

## 关键改动

```text
python/sglang/utils.py: TypeBasedDispatcher._mapping 改 OrderedDict;__call__ 三级查找(精确 dict / MRO 缓存 / isinstance 遍历+缓存)
test/srt/test_bench_typebaseddispatcher.py: 新增基准脚本(list 对照、继承、内存、边界)
test/srt/test_type_based_dispatcher.py: 新增真实 io_struct 请求类型的 E2E 派发测试
test/srt/run_suite.py: 注册两个新测试文件
```

```python
def __call__(self, obj: Any):
    obj_type = type(obj)
    # 1. First try exact match(o(1))
    fn = self._mapping.get(obj_type)
    if fn is not None:
        return fn(obj)
    # 2. If exact match fails, check MRO cache
    cached_fn = self._mro_cache.get(obj_type)
    if cached_fn is not None:
        return cached_fn(obj)
    # 3. search in registration order for compatible type
    for ty, fn in self._mapping.items():
        if isinstance(obj, ty):
            self._mro_cache[obj_type] = fn
            return fn(obj)
```

## 性能声明

内部基准:30 种请求类型、每测试 1000 次 dispatcher 调用(未注明硬件与精度设置,PR 未注明测试条件中的机器信息):

| 场景 | List 版 | Dict 版 | 提升 |
| --- | ---: | ---: | ---: |
| First Element | 0.0059 s | 0.0043 s | 1.37x |
| Middle Element | 0.025 s | 0.0043 s | 4.78x |
| Last Element | 0.030 s | 0.0043 s | 6.99x |
| Random Element | 0.048 s | 0.0043 s | 3.51x |
| Mixed Requests(模拟真实 workload) | 0.3474 s | 0.0436 s | 7.97x |

更新后的结论:精确匹配 3-7x、首次继承匹配 1-3x、缓存后的继承匹配 4-8x。收益随注册类型数增长,原 list 位置越靠后收益越大。

## 验证与回归风险

不涉及模型输出与 kernel,正文称所有既有单测未修改即通过;新增边界用例(未知类型、fallback、继承、内存对比)与真实 io_struct 请求类型的 E2E 测试。风险:继承派发依赖 MRO 缓存正确失效(`__iadd__` 已清缓存);注册顺序语义靠 OrderedDict 维持,若多个注册类型同时 isinstance 匹配,仍按注册顺序首个命中,行为与原实现一致;该 dispatcher 是 scheduler/tokenizer 间消息派发的公共路径,属于全模型无关的基础设施改动。

## 关联

无(纯基础设施优化;其他同分类 PR 编号素材未注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
