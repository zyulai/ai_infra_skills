# PR #1395: fix RAM OOM when load large models in tensor parallel mode.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#1395](https://github.com/vllm-project/vllm/pull/1395) |
| 作者 | boydfd |
| 合并日期 | 2023-11-21 |
| labels | (none) |
| 热度 | 👍 2 · 💬 9 |
| 分类 | Parallelism / Communication |
| bug 类型 | resource |

## 一句话结论
TP 加载大模型时所有 worker 并发加载撑爆节点 RAM;新增 --max-parallel-loading-workers 把 worker 分批加载。

## 问题现象
tensor parallel 模式加载大模型时 RAM OOM:TP=8 启动 WizardLM--WizardCoder-15B-V1.0 时 Ray 杀掉 worker(node 内存 242.20GB / 251.71GB,超 0.95 阈值;8 个 RayWorker 各占约 29.4-29.6GB,报 `ray.exceptions.OutOfMemoryError`);加 `--tensor-parallel-model-load-batch-size 2` 后启动成功(#GPU blocks: 31424)。fix bug: #322 #872。

## 根因
正文未披露根因(仅给出「fix bug: #322 #872」与测试日志)。(由 diff 推断)此前模型加载在 `init_model` 中,所有 worker 同时调用 `get_model` 并发加载,每个进程承受完整权重加载的内存峰值,TP 越大同时加载的进程越多、节点 RAM 峰值越高;修复按 `max_parallel_loading_workers` 把 worker 分批,限制同时加载的进程数。

## 修复方案
1. `arg_utils.py` 新增 CLI `--max-parallel-loading-workers`,存入 `ParallelConfig`;
2. `llm_engine.py`:`_init_workers`/`_init_workers_ray` 把 `load_model` 拆成独立调用并传 `max_concurrent_workers`;`_run_workers` 重构出 `_run_workers_in_batch`,按 work_groups 分批执行;
3. `worker.py`:新增 `load_model()`,`init_model` 不再加载模型。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:engine 的 worker 执行路径重构(`_run_workers`/`_run_workers_in_batch`);启动流程拆出 `load_model` 阶段;默认 `max_parallel_loading_workers=None` 时全部 worker 一批,行为与原先一致(由 diff 推断);分批加载会拉长启动时间(换 RAM 峰值下降)。

## 关联
- fix bug: #322 #872

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
