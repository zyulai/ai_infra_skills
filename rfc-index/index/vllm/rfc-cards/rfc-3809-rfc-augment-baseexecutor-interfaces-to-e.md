# RFC #3809: [RFC] Augment BaseExecutor interfaces to enable hardware-agnostic speculative decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3809](https://github.com/vllm-project/vllm/issues/3809) |
| 作者 | cadedaniel |
| 创建日期 | 2024-04-03 |
| 状态 | ⚪ closed(关闭于 2024-04-09) |
| labels | RFC |
| 热度 | 👍 3 · 💬 7 |
| 分类 | Speculative Decoding |

## 一句话结论
提议扩充 `ExecutorBase`/`WorkerBase` 接口(init_workers、init_cache、profile_num_available_blocks),使投机解码逻辑运行在 Worker 之上、与硬件后端无关。

## 提案动机
投机解码的核心逻辑是算法层面的 vLLM 特性,不应因后端不同而各有实现。投机解码都由三个阶段组成:propose tokens、score tokens、verify tokens;propose 与 verify 均有多种实现(草稿模型可换成 prompt-lookup-decoding,典型接受可换成 rejection sampling)。这一层抽象意味着投机解码框架可以实现在 Worker 层之上,让非 GPU worker 也能纳入框架——本 RFC 就是提出所需的接口。作者还举了同层特性的例子:jump decoding(对可 100% 确定的输出片段跳过生成,vLLM 当时未实现),它可视为投机解码的特例,留作 future work / 超出范围。

## 提出的方案
- 提出统一的 `WorkerBase(ABC)` 接口,包含:`init_device`、`profile_num_available_blocks`(返回 `tuple[num_device_blocks, num_cpu_blocks]`,device 指活动 KV cache、cpu 指被动/swap cache;GPUExecutor 返回 `[num_gpu_blocks, num_cpu_blocks]`,未来的 CPUExecutor 可返回 `[num_cpu_blocks, 0]` 等)、`init_cache`(与 init_workers 分开,因为需要先 profile 决定 KV cache 上限)、`execute_model`、`add_lora`/`remove_lora`、`check_health`。
- 给 `ExecutorBase` 增加三个抽象方法:`init_workers`、`profile_num_available_blocks`、`init_cache`(以 diff 形式给出)。
- 现有实现需相应调整:`GPUExecutor` 与 `RayGPUExecutor` 的 profiling 编排逻辑上移回 `LLMEngine`;`NeuronExecutor` 覆写 `profile_num_available_blocks` 返回 `[max_num_seq, 0]`。

## 优化点与收益
正文未给出量化数字。收益为架构性:投机解码在 Worker 层之上统一实现,所有 worker(GPU、Neuron、未来 CPU 等)自动兼容投机解码;避免同一算法特性在不同后端重复实现。

## 现状与落地
已关闭(2024-04-09,创建后仅 6 天)。正文给出接口草案与实现迁移要点;落地情况需查关联 PR(此接口后续随 vLLM 重构演进,与现状的对应关系需另行核实)。

## 关联
- Jump decoding 博客: https://lmsys.org/blog/2024-02-05-compressed-fsm/#our-method-jump-forward-decoding-with-a-compressed-finite-state-machine

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
