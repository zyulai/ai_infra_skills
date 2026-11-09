# RFC #18009: [RFC]: Model Parallelism with Single Worker using SPMD

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18009](https://github.com/vllm-project/vllm/issues/18009) |
| 作者 | lsy323 |
| 创建日期 | 2025-05-12 |
| 状态 | ⚪ closed (关闭于 2025-11-13) |
| labels | RFC, unstale |
| 热度 | 👍 11 · 💬 7 |
| 分类 | Architecture / Engine |

## 一句话结论
提议为 TPU 增加单 worker 的 SPMD 模型并行:模型按单设备编写,编译器(GSPMD)依 hint 自动切分并插入通信。

## 提案动机
当前 vLLM 多芯片并行是每设备一个 worker,reduce、gather 等通信算子显式写在模型实现里;而 PyTorch/XLA 与 GSPMD 提供另一种编程模型:模型如同跑在单设备上编写(无显式通信 op,代码保持 global shape),通过编译器 hint 让编译器自动插入 collective 并做通信/计算重叠优化。这类只有编译 pass 才能开启的高级优化对 TPU 多芯片推理性能至关重要;同时模型代码与并行配置解耦,给开发者提供另一种并行实现方式(类似 torchtitan)。

## 提出的方案
- 单 worker 模式:一个进程管理同 host 全部设备并执行同一编译后程序;多 host 时每 host 一个进程;用环境变量启用(作者计划将来把单 worker 作为 TPU 默认分布式后端)
- 扩展 executor 类型判定逻辑:单 worker 模式使用单进程 executor
- 模型初始化/加载:Meta device 初始化全模型(不占物理内存)→ pre-hook 把权重移到 CPU 加载 → post-hook 应用编译 hint 切分权重并传到 TPU;避免在 TPU 上 init/load 触发图编译、占用额外内存(host OOM,如 DeepSeek v3 超过 1.2 TB)
- 新增 vllm/distributed/tpu_distributed_utils.py:为 RowParallelLinear、QKVParallelLinear 等并行层附加编译 hint,再递归处理模型各子模块
- module swap:用 SPMD 友好的实现替换 QKVParallelLinear(q/k/v 权重分开存储;合并的大 tensor 切分会产生低效计算图)
- KV cache 切分在 TPU model runner 创建并绑定 cache tensor 时打 annotation;num_kv_heads < tp_size 时保证每 partition 至少 1 个 kv_head;torch.compile 在 runner 内重新套 wrapper;profiling/warmup/execute_model 与 scheduler 均无需改动

## 优化点与收益
正文未给出量化数字。定性收益:解锁仅编译 pass 可实现的 TPU 多芯片优化;模型代码不再与并行配置耦合。

## 现状与落地
已关闭。正文附有原型代码链接(作者 fork),反馈期 2 周;正文未写明最终结果,落地情况需查关联 PR。

## 关联
- GSPMD 论文:https://arxiv.org/pdf/2105.04663
- PyTorch/XLA SPMD 指南:https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_basic.md
- FFN SPMD 示例:https://github.com/lsy323/learning_machine/blob/lsiyuan/torch-xla-example-vllm/torchxla-for-vllm/ffn_spmd.py#L57-L60

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
