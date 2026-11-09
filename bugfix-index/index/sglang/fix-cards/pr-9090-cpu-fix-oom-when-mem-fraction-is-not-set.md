# PR #9090: [CPU] fix OOM when mem-fraction is not set

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9090](https://github.com/sgl-project/sglang/pull/9090) |
| 作者 | ZailiWang |
| 合并日期 | 2025-09-11 |
| labels | high priority, ready-to-merge, intel, cpu |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
CPU 沿用加速卡默认 mem-fraction-static=0.88 致 OOM;可用内存改为按 NUMA 节点均分并加 BIND 告警。

## 问题现象
正文描述:`server_args.py` 在命令行未设置时把默认 `mem-fraction-static` 定为 0.88(正文链接指向 server_args.py#L354),这对加速器设备适用,但 CPU 服务器上内存被所有 TP rank 共享。CPU 上不设置 `--mem-fraction-static`,或简单设为(如 0.8、0.9)且未指定 `--max-total-tokens` 时,可能 OOM。

## 根因
正文披露:默认 0.88 按加速卡(显存独立)假设设计,CPU 场景下内存由所有 TP rank 共享,沿用该默认或简单比例分配会超出实际可用内存,导致 OOM。另因 `SGLANG_CPU_OMP_THREADS_BIND` 设置后各 rank 可用内存无法预知,正文与代码补充告警提醒用户自行负责内存管理。

## 修复方案
- `python/sglang/srt/utils.py` `get_available_gpu_memory` CPU 分支:可用内存不再取 psutil 全机 `available`,改为 `total_free_memory / n_numa_node`(按 NUMA 节点数均分,保留 3 位小数);正文说明每 NUMA 节点内存查询最初计划用 py-libnuma,因其不兼容 py311/py313 改为直接调用 `libnuma.so` 的 API。
- `python/sglang/srt/model_executor/model_runner.py` `init_threads_binding`:无论 `SGLANG_CPU_OMP_THREADS_BIND` 是否设置都计算 `cpu_ids_by_node`/`n_numa_node`;BIND 设置时新增 assert 要求列表数与 tp_size 一致,并在 tp_size 大于 NUMA 节点数时告警提示设置 `--max-total-tokens`。
- 配套:`docs/platforms/cpu_server.md` 补充告警并调整示例;`Dockerfile.xeon` 与 `python/pyproject.toml` 把 intel-openmp 从 Dockerfile 移到 `srt_cpu` extra、去掉 `-v`;测试 `test_intel_amx_attention_backend.py` 的 mem-fraction 由 0.05 调到 0.3。

```diff
     elif device == "cpu":
         # TODO: rename the variables in the current function to be not GPU specific
-        free_gpu_memory = psutil.virtual_memory().available
+        total_free_memory = psutil.virtual_memory().available
+        n_numa_node: int = len(get_cpu_ids_by_node())
+        free_gpu_memory = round(total_free_memory / n_numa_node, 3)
```

## 影响与回归风险
- 回归面:仅 CPU 后端:上报的可用内存变小(按 NUMA 均分),KV cache 等预算更保守;`SGLANG_CPU_OMP_THREADS_BIND` 现在要求列表数与 tp_size 严格相等,不满足会直接 assert 失败(原先不校验);tp_size 大于 NUMA 节点数时只告警不阻断。`Dockerfile.xeon` 与 `srt_cpu` extra 的依赖调整影响 CPU 镜像与安装流程(intel-openmp 改由 extra 提供)。GPU/NPU 路径不变。

## 关联
正文未列出(仅链接了 server_args.py 代码位置,无 issue/PR 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
