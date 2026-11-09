# PR #36164: perf: add __slots__ to KVCacheBlock

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cong-or |
| 合并 | 2026-03-06 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 2 个文件(实际源码改动 1 行) |
| 链接 | https://github.com/vllm-project/vllm/pull/36164 |

## 一句话结论

给 `KVCacheBlock` dataclass 加 slots,实例从 344 字节缩到 80 字节,10 万块分配快 1.54 倍并省约 25 MB。

## 动机

`KVCacheBlock` 是 KV cache 路径中分配最多的 Python 对象——每个块一个实例,典型部署超过 10 万个。作为普通 `@dataclass`,每个实例都背着一个并不需要的 `__dict__` 哈希表:既放大内存占用,又增加分配耗时与 GC 压力,而块元数据的分配发生在调度关键路径上。

## 优化点

机制是最小化 CPython 对象布局:把 `@dataclass` 改为 `@dataclass(slots=True)`,生成的类用固定槽位替代每实例 `__dict__`。安全性由三个事实保证(PR 正文核实):没有任何代码对这些对象动态设置属性、没有类继承它、对象从不被 pickle。改动本身一行,运行时行为(字段访问、链表操作)完全不变,收益纯粹来自内存布局:每实例 344 → 80 字节,分配耗时降约 35%。

## 关键改动

```text
vllm/v1/core/kv_cache_utils.py: KVCacheBlock 改为 @dataclass(slots=True)
tests/v1/core/test_kv_cache_utils.py: 新增 slots 行为守卫测试
```

```python
-@dataclass
+@dataclass(slots=True)
 class KVCacheBlock:
     """KV-cache block metadata."""
```

## 性能声明

独立 benchmark 分配 10 万个 block(未注明模型/硬件):每实例 **344 → 80 字节(小 77%)**;10 万块合计**省约 25 MB**;分配速度 **1.54x**。为微基准数字,非端到端引擎测试。

## 验证与回归风险

新增 `test_kv_cache_block_uses_slots`:断言实例无 `__dict__`、动态属性赋值抛 `AttributeError`;既有测试覆盖全部字段访问与 free/parking 链表操作。风险点:任何依赖动态属性的下游代码(如 monkey-patch、调试工具)会开始报错;若未来有子类需要自己的 slots 需要显式声明。属于低风险的纯内存布局改动。

## 关联

同区域(调度器/缓存元数据对象开销):#21917(优化 request 移除)、#21146/#24128(CUDA graph 捕获期间冻结/延后 GC)、#23674(用 sha256 bytes 替代 BlockHash 开销路径)。
