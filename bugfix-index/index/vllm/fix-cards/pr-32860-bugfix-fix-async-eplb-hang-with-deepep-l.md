# PR #32860: [BugFix] Fix async EPLB hang with DeepEP LL all2all backend

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32860](https://github.com/vllm-project/vllm/pull/32860) |
| 作者 | ilmarkov |
| 合并日期 | 2026-02-10 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | hang |

## 一句话结论
async EPLB 的 NCCL 权重交换占满 SM 阻塞 DeepEP LL 协作启动,跨 rank 交错成死锁;限制 NCCL_MAX_CTAS=8 修复。

## 问题现象
同时启用 async EPLB 与 DeepEP low-latency all2all 后端时挂起(hang)。验证场景:DP/EP 模式(-tp 1 -dp 4)下服务 `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`,`--async-scheduling --enable-eplb` 且 `--eplb-config.use_async true`;修复后 gsm8k 评测通过(exact_match flexible-extract 0.8810)。

## 根因
正文披露:挂起源于 DeepEP LL 的 cooperative launch。async EPLB 在异步线程中为权重交换启动 NCCL kernel,占用大量 SM,从而阻塞 DeepEP kernel;当这些 kernel 在不同 rank 上乱序启动时形成死锁。具体机制(详见新增代码注释):两个 rank 在 NCCL 集合通信(async EPLB 权重交换)与 DeepEP LL kernel 之间以不同顺序交错启动——DeepEP LL 协作启动要预留大部分 GPU SM,若 SM 被 NCCL 占用则阻塞等待;rank A 在主线程进入 DeepEP LL 等 SM,rank B 在异步线程的 NCCL 里等 rank A 参与集合通信,循环等待即死锁。参见 DeepEP issue #496。

## 修复方案
新增 `vllm/distributed/eplb/eplb_utils.py` 的 `override_envs_for_eplb`:当 data_parallel_size > 1、启用 EPLB、all2all 后端为 `deepep_low_latency` 且 `eplb_config.use_async` 为 true 时,若用户未显式设置 `NCCL_MAX_CTAS`,则将其覆盖为 8,为 DeepEP 协作 kernel 留出 SM,打破死锁;并在 `gpu_worker.py` 的 `init_worker_distributed_environment` 中(初始化 batch invariance 之后、custom all-reduce 之前)调用。

关键 diff 摘录:
```python
if is_data_parallel and is_eplb_enabled and is_deepep_ll and async_eplb:
    current_value_str = os.getenv("NCCL_MAX_CTAS")
    if current_value_str and current_value_str.isdigit():
        return
    override_value = 8
    os.environ["NCCL_MAX_CTAS"] = str(override_value)
```

## 影响与回归风险
- 回归面:仅新增分布式环境初始化时的环境变量覆盖;命中条件限定为 DP/EP + EPLB + DeepEP LL + async EPLB。正文说明 DP/EP 模式热路径不使用 NCCL,因此该限制不直接影响性能;用户已显式设置 NCCL_MAX_CTAS 时不会被覆盖。

## 关联
- [DeepEP issue #496](https://github.com/deepseek-ai/DeepEP/issues/496)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
