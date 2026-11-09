# PR #30800: [PERF] Add interleaved memory allocation to NUMA module

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @skaraban3807 |
| 合并 | 2025-12-24 |
| 分类 | KV Cache / Memory \| tier: major |
| Label | ready, cpu |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/30800 |

## 一句话结论

多 NUMA 节点时改用交错内存分配，AMD EPYC 内存密集负载平均约 30% 性能提升。

## 动机

PR 正文 Purpose 说明：系统配置多于一个 NUMA 节点时新增 interleaved memory allocation，使内存流量在节点间分布更均匀，对 memory-bound workloads 尤其有利。正文称在受测 AMD EPYC 系统上为 memory-bound workloads 带来约 30% 性能提升。

## 优化点

改动集中在 CPU 后端的 NUMA 绑定逻辑 `csrc/cpu/utils.cpp`：删除原先只取首个 CPU 所在 NUMA 节点（`mem_node_id`）并据此绑定内存、以及跨节点告警的逻辑，改为收集所有 `node_ids`。将 `numa_get_membind()` 替换为 `numa_get_mems_allowed()` 作为 src_mask。当 `node_ids.size() > 1` 时调用 `numa_set_interleave_mask(mask)` 采用 INTERLEAVE 策略，把内存分配交错到所选多个 NUMA 节点；单节点时仍用 `numa_set_membind(mask)` 的 MEMBIND 策略。同时对 `numa_parse_cpustring_all` 返回空指针增加 `TORCH_CHECK` 校验。

## 关键改动

```text
csrc/cpu/utils.cpp: NUMA 内存绑定按节点数切换 INTERLEAVE / MEMBIND 策略，并补充空指针校验
```

```text
-      bitmask* src_mask = numa_get_membind();
+      bitmask* src_mask = numa_get_mems_allowed();
+        if (node_ids.size() > 1) {
+          numa_set_interleave_mask(mask);
+        } else {
+          numa_set_membind(mask);
+        }
```

## 性能声明

正文 Test Plan：`VLLM_CPU_KVCACHE_SPACE=100`、`VLLM_CPU_OMP_THREADS_BIND=0-95`、`LD_PRELOAD` 加载 libtcmalloc_minimal 与 libiomp5，命令 `vllm bench throughput --input-len 128 --output-len 1024 --model <model_name> --num-prompts 96 --dtype bfloat16 --max-num-seqs 32`。Test Result："I have validated this change on AMD EPYC servers with NPS4-NUMA configurations. Across a range of memory-intensive benchmarks, I observed an average performance improvement of approximately 30% compared to the baseline (non-interleaved) configuration." 正文未给出逐项数值，模型名以 `<model_name>` 占位。

## 验证与回归风险

正文给出上述 bench throughput 命令与 AMD EPYC NPS4-NUMA 上的验证结论。Checklist 中 Purpose / Test Plan / Test Result 三项勾选，文档与 release notes 更新项未勾选。改动仅影响 CPU 平台、存在多 NUMA 节点时的内存绑定路径；正文未说明对单 NUMA 节点、其他硬件或精度的影响，也未给出新增测试。正文 Test Result 另附一张性能对比截图链接。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
