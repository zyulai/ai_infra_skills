# PR #68162: [Tests] Reuse single-rank parallel groups for faster GPU unit tests

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd(共同作者 Tomas Ruiz @tomasruizt) |
| 合并 | 2026-11-18 |
| 分类 | MoE(tier: major) |
| Label | ready, ci/build |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/68162 |

## 一句话结论

把共享专家测试里既有的单 rank 并行组替身提取为 `tests/single_rank_group.py`,在量化、shared-expert、routing、expert-substitution、zero-expert 等测试中替代 `dist_init` 与逐项 TP patch,并配合 `skip_global_cleanup` 去掉重复 teardown——实测的 33 个 GPU 单测选择集从 487.79 s 降到 190.87 s(约 2.6x),无生产代码改动。

## 动机

单 GPU 层测试在并行性无关时不需要真正的分布式进程组:只需要 rank 0 / world size 1 / identity collectives 的元数据。main 上这样的测试要付两份 cleanup:`dist_init` fixture 的 `yield` 后调 `cleanup_dist_env_and_memory()`(完整 `gc.collect()`,约 0.4 s / 约 100 万活对象),autouse 的 `cleanup_fixture` 再调一次——每次一个 full `gc.collect()`。替身已在 shared-expert 测试里存在(`tests/model_executor/layers/test_fused_shared_expert.py` 的 `_SingleRankGroup`),只是没有共享出来。

## 优化点

- 新文件 `tests/single_rank_group.py`:`SingleRankGroup`(rank 0、world 1、all_reduce/all_gather/reduce_scatter 等恒等 collectives、无 FileStore/backend、不初始化 model-parallel group)+ `override_parallel_groups(group)` 装饰器——替换 `_WORLD/_TP/_ETP/_DCP/_PP/_DP/_EP/_PCP` 八个组,返回或异常时全部恢复;支持 yield fixture(对生成器函数包 `yield from`),pytest 签名保持。
- 采纳方式只加装饰器/`skip_global_cleanup` 标记、删除过时的单 rank 设置;测试体、断言、容差、平台 skip 不变;TP>1 模拟 patch 保留。装饰器函数打 `_uses_single_rank_groups` 标记,MoE 的 hook 认得它以保留 clean-entry 隔离;online FSE 模块保留原 cleanup(模块级 skip 曾导致其 checkpoint-weight 相等测试失败)。
- CI:Quantization CUDA/ROCm、CUDA Quantized Models、AMD model-executor、MoE CUDA/ROCm 等 job 的 `source_file_dependencies` 加入 `tests/single_rank_group.py`。

## 关键改动

```python
# tests/single_rank_group.py(新文件,pack diff 摘录)
_GROUP_NAMES = ("_WORLD", "_TP", "_ETP", "_DCP", "_PP", "_DP", "_EP", "_PCP")

class SingleRankGroup:
    """No-op single-rank stand-in for a compiled `GroupCoordinator`."""

    world_size = 1
    rank_in_group = 0
    local_rank = 0
    is_first_rank = True
    is_last_rank = True

    def all_reduce(self, input_: torch.Tensor, *args: Any, **kwargs: Any):
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = 0):
        return input_
```

```diff
-def test_standard_fused_moe_model_discovers_substitution(dist_init):
+@override_parallel_groups(SingleRankGroup())
+def test_standard_fused_moe_model_discovers_substitution():
```

```diff
 # tests/kernels/moe/conftest.py:hook 认得装饰过的测试
         or "dist_init" in item.fixturenames
+        or getattr(item.obj, "_uses_single_rank_groups", False)
     ):
```

## 性能声明

- 全 PR 在实测选择集(33 个测试文件/用例,301 passed / 63 skipped,三种配置各一次、同一设备依次运行)上比 pinned main baseline 快约 2.6x:487.79 s → 190.87 s。分项:main setup 31.05 s / body 118.69 s / teardown 335.56 s;仅装饰器(不加 `skip_global_cleanup`)352.25 s(teardown 210.61 s);装饰器 + 新增 `skip_global_cleanup` 190.87 s(teardown 49.01 s)。
- 单例(`test_online_quantization_targets[linear_regex]`,7 次中位数):main 1513 ms(cleanup_dist_env_and_memory 2 次)→ 仅装饰器 771 ms(1 次)→ 再加 skip 365 ms(0 次)。
- 环境:AMD Instinct MI355X、ROCm/torch HIP 7.15.26333、AMD clang 23.0.0git;main pinned 到分支 base commit fa6c77f5954e;PR 两行均测 ad4f2308116f。作者注明计时噪声大、不是普遍性能保证;装饰器本身的聚合增益小于示例(301 个通过测试中仅 131 个在 main 用 `dist_init`),而 cleanup skip 给出一致的约 2.5x。

## 验证与回归风险

- 直接 helper 检查覆盖 collective identity 与嵌套/装饰器异常恢复,且不初始化 `torch.distributed`;两 GPU 启动验证未含在此 PR(其目标是单 rank setup,不是分布式正确性)。CUDA/XPU 计时与多 rank 验证留给其他 PR。
- 风险(正文明示):不模拟真实通信或异步 collectives;aggregate 增益依赖测试选择,run-to-run 变化大(另一设备早前一轮 main 339 s vs 无额外 cleanup skip 的 336-363 s,在噪声内)。

## 关联

- 从 #67529 抽出为独立 PR;正文明示未找到其他实现相同思路的 open PR;改为 test-only,无生产代码或用户可见行为变化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
