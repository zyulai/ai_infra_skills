# PR #52372: [Bugfix][Mooncake] Reference GPU blocks for in-flight store jobs and key the store ledger by store_job_id

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chengy-sysu |
| 合并 | 2026-08-16 |
| 分类 | Other(major,KV connector) |
| Label | bug, ready, kv-connector |
| Diff | diffstat 缺失(diff 全文已提供但裁剪了 3 个文件,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/52372 |

## 一句话结论

修复 Mooncake KV 异步存储两个正确性 bug:store job 读到已释放/被覆写的 GPU block 造成 KV 污染,以及 preemption 后 req_id 复用导致账本错乱、block 永久滞留。

## 动机

Mooncake store 的 KV 上传是异步的:step 0 只把请求排入队列,GPU 读取与 PUT 由后台线程稍后执行;从 step 1 起调度器可以抢占该请求并立即把其 block 归还 free pool 供其他请求覆写(worker 要到下一次 IPC 才知道)。此窗口内发出的 PUT 读到的是别的请求的 KV;由于 key 命名固定 token 范围,污染数据不会自愈,后续所有命中同前缀的请求都会读到错 KV,影响面随时间扩大。既有唯一保护 `request_finished`→`delay_free_blocks` 只覆盖正常完成路径,preemption 路径(`build_connector_meta` 重置 tracker 后立即释放)完全无保护。第二个 bug:#51637——被抢占请求恢复时沿用原 req_id,store 线程按 req_id 计数的账本无法区分新旧代际,遗留 job 会错误递减新代计数,最坏导致计数为负、条目永不退休、`finished_sending` 不再发出、block 永不释放,引擎 KV 饥饿。

## 优化点

让每个 store job 自己在 GPU block pool 上持引用:`store_job_id`(调度器侧单调递增计数器)唯一标识一个 job,scheduler 在 `build_connector_meta` 时对 job 覆盖的 block(`_pinned_saves[store_job_id]`,含 partial-tail offload 读取的、故意不在请求 block 表内的 CoW block)增加引用,等所有 rank 经 `MooncakeStoreWorkerMetadata.completed_saves`(store_job_id→已报告 rank 数)聚合上报完成后释放;期间 `has_pending_push_work()` 驱动引擎持续 step。由此 `request_finished` 不再需要延迟释放(直接返回 `(False, None)`),worker 也不再上报 `finished_sending`。第二处:store 线程账本从"每 req_id 一个计数"改为 `req_id -> {live store_job_ids}`,遗留 job 不在新代集合内,退休改为"集合为空",下溢结构性不可能;遗留 job 也不能 rewind 新代 `_saved_offset` 或标记 skip。

## 关键改动

```text
vllm/.../mooncake/store/scheduler.py: _pinned_saves 引用计数、store_job_id 分配、has_pending_push_work、request_finished 移除
vllm/.../mooncake/store/data.py: ReqMeta.store_job_id;MooncakeStoreWorkerMetadata(completed_saves 聚合)
vllm/.../mooncake/store/connector.py: bind_gpu_block_pool / has_pending_push_work / build_connector_worker_meta 接线
vllm/.../mooncake/store/worker.py: job 生命周期(finish_store_job/is_live_store_job),store 抛异常也保证上报(3 个文件被裁剪)
```

completed_saves 聚合:

```python
class MooncakeStoreWorkerMetadata(KVConnectorWorkerMetadata):
    completed_saves: dict[int, int]
    def aggregate(self, other):
        for store_job_id, count in other.completed_saves.items():
            self.completed_saves[store_job_id] = (
                self.completed_saves.get(store_job_id, 0) + count)
        return self
```

## 性能声明

修复后无性能开销的 A/B(基线 `64ca614fe`,无 instrumentation):wall 54.5s→55.2s(+1.3%),TTFT p50 9.84s→8.62s、p90 13.41s→13.39s,preemptions 165→192,external prefix cache lookups 两臂完全相同(134,777,证明负载确定、可直比);早期四次运行中修复侧反而快 2-3.5%。PR 结论为"无可测开销"。正确性验证(非性能):H20 TP2、Qwen3-32B-FP8、`--num-gpu-blocks-override 1024`、96 请求/48 并发/冷 store,指纹错配 588→0、被污染 key 438→0、读错 KV 的请求 8/96→0/96,修复侧仍发生 125 次 preemption 但不再产生错 KV。

## 验证与回归风险

新增 5 个针对性单测(引用在全部 rank 上报后才释放、CoW block 引用、跨 rank 完成聚合、遗留 job 无法触碰复用 req_id、store 抛异常仍上报防泄漏);10 个 mooncake 测试文件 251 passed / 16 failed(失败均为无 GPU 环境的 P2P transfer-engine 文件,本 PR 未触碰);ruff 通过。风险面:引擎需在 PUT 在途时保持 step(`has_pending_push_work`),preemption 计数变化;与 #51662(同问题另一修法,job.epoch)互斥,二选一即可。

## 关联

FIX #52360、#51637;相关:#51662(替代修法)、#43742、#51595、#43226/#46240(均为相邻的异步完成处理问题)。KV connector 区域其他 PR 编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
