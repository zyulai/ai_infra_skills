# RFC #33689: [RFC]: KV Offloading Roadmap

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33689](https://github.com/vllm-project/vllm/issues/33689) |
| 作者 | orozery |
| 创建日期 | 2026-02-03 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 14 · 💬 2 |
| 分类 | CI / Infra / Process |

## 一句话结论
KV offloading 的滚动路线图:盘点已支持的卸载/加载特性并规划策略抽象等后续工作。

## 提案动机
正文没有独立的 Motivation 叙述章节,直接以特性清单(roadmap)形式组织,如实说明:全文为「Currently supported / Upcoming / Backlog / Tactical tasks」四段 checklist,用于跟踪 KV offloading 子系统的现状与后续任务,未附背景论证或量化论证。

## 提出的方案
(勾选状态照抄正文)
- **Currently supported(全部已完成)**:可插拔的 out-of-tree 卸载后端;CPU-GPU Offloading(NVIDIA+AMD);自定义 offloading block size;完全异步(跨 engine step)的 offloading/loading;Immediate Offloading(与 spill-over 相对);LRU Eviction;跨层块;Metrics;把被抢占请求 onload 回来;KV Events;Kernel-block-size 支持;抽象的淘汰策略(LRU、ARC 等);HMA Support。
- **Upcoming**:Documentation(已完成);把跨层块扩展到 hybrid 模型(未完成);支持分层存储 tiering,如 CPU -> 可插拔后端(已完成)。
- **Backlog**:创建卸载策略抽象,含两种模式——Preemption-only offload mode(旧 PR #32398)与 Spillover mode(swap-in swap-out)。
- **Tactical tasks(代码重构清单,节选)**:文件搬迁与重命名(`worker/cpu_gpu.py` → `cpu/gpu_worker.py`、`abstract.py` → `base.py`、`reuse_manager.py` → `cpu/manager.py` 等);用 `OffloadingWorker`(抽象函数 `submit_store`/`submit_load`)替换 `OffloadingHandler`;为 `OffloadingManager` 增加 `request_finished` 方法;offloading connector 偏好 HND layout;支持 `reset_cache`;把硬件无关部分从 `cpu/gpu_worker.py` 移到 `cpu/worker.py`;CPU 内存 pinning 移到后台线程(此两项未完成)等。

## 优化点与收益
正文未给出量化数字。收益为架构与功能层面:以统一路线图推进 KV offloading 的后端可插拔、异步传输、淘汰策略抽象与分层存储,降低不同硬件/后端组合下的维护成本。

## 现状与落地
持续推进中:状态 open(labels 含 keep-open);已支持清单与多数 tactical 任务已勾选完成,剩余项为 hybrid 模型跨层支持、offload 策略抽象(Backlog)及部分重构任务。

## 关联
- PR [#32398](https://github.com/vllm-project/vllm/pull/32398)(Backlog 中引用的旧 Preemption-only offload PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
