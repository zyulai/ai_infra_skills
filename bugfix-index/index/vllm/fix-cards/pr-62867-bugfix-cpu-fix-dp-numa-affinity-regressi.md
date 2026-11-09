# PR #62867: [Bugfix][CPU] Fix DP NUMA affinity regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62867](https://github.com/vllm-project/vllm/pull/62867) |
| 作者 | bigPYJ1151 |
| 合并日期 | 2026-10-23 |
| labels | bug, cpu |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#45026 重构后 CPU DP 所有 rank 绑到同一 NUMA 节点;让 `get_visible_memory_node()` 优先使用按 rank 分配的节点 id,各 DP rank 绑回各自节点。

## 问题现象
在多 NUMA 节点的 CPU 宿主机上使用 `--data-parallel-size N` 时,每个 `EngineCore_DP*` 都绑定同一组核:6 NUMA 节点主机上 `vllm serve facebook/opt-125m --data-parallel-size 2 --max-model-len 256`,DP0 与 DP1 的 OpenMP 绑定均为 cores 192-222(同一节点)。对应 issue #47328。修复后 DP0 绑 node 0(cores 192-222)、DP1 绑 node 1(cores 224-254),内存分别 MEMBIND 到节点 0/1,端到端请求返回正确。

## 根因
PR 正文披露:#45026 把每个 DP rank 的设备列表从设备控制环境变量挪进 `parallel_config.assigned_physical_gpu_ids`;`CoreEngineProcManager` 仍计算并写入各 rank 分片,GPU/XPU 也消费它,但 CPU 代码仍只读 `CPU_VISIBLE_MEMORY_NODES`——已没有任何代码按 rank 设置它。于是 EngineCore 进程中的 `OMPProcessManager` 看到完整节点列表,`_get_autobind_cpu_ids` 取 `allowed_numa_nodes[local_rank]`,所有 DP rank 都选到同一个节点。

## 修复方案
`vllm/utils/cpu_resource_utils.py` 的 `get_visible_memory_node()` 增加可选参数 `assigned_node_ids`,优先级为:assigned ids > `CPU_VISIBLE_MEMORY_NODES` > 全部节点;`VLLM_CPU_SIM_MULTI_NUMA` 仍覆盖一切:

```diff
-def get_visible_memory_node() -> list[int]:
+def get_visible_memory_node(assigned_node_ids: list[int] | None = None) -> list[int]:
...
+    if assigned_node_ids is not None:
+        return [node for node in assigned_node_ids if node in allowed_memory_node_list]
```

三个调用点传入分配的节点 id:`OMPProcessManager`(读 config;进程内 global 只在 worker 进程里设置,EngineCore 进程拿不到)、`CPUWorker`(读 `vllm_config.parallel_config.assigned_physical_gpu_ids`)、`CpuPlatform.get_current_memory_usage`(worker 内用 global)。测试方面新增按 rank 分配节点、覆盖 env、限制在 allowed 集合内、SIM 开关忽略 assigned ids 等用例,及 `test_omp_manager_binds_each_dp_rank_to_its_assigned_node`(两个 DP rank 得到不相交核心,若忽略 assigned ids 则失败)。

## 影响与回归风险
- 影响窗口:自 #45026 起 CPU DP 的 NUMA 绑定受影响。
- 回归面:仅 CPU 路径(约 26 行新增/10 行删除,不含测试),不动共享 engine 代码;不带 DP 的手动 `CPU_VISIBLE_MEMORY_NODES` 行为不变。正文指出的已知副作用:`--device-ids` 也会填 `assigned_physical_gpu_ids`,CPU 上它现在会选择 NUMA 节点(与 `CPU_VISIBLE_MEMORY_NODES` 组合),与 GPU 一致,但该路径未被上述验证覆盖。单测 `tests/utils_/test_cpu_resource_utils.py` + `tests/v1/worker/test_cpu_worker_kv_budget.py` 21 passed。

## 关联
- #45026(引入回归的 PR)
- issue #47328(被修复的 issue)
(正文以编号引用,未给出完整链接。)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
