# RFC #16144: [RFC]: Offload KV cache to CPU in V1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#16144](https://github.com/vllm-project/vllm/issues/16144) |
| 作者 | mengzhu28 |
| 创建日期 | 2025-04-07 |
| 状态 | ⚪ closed (关闭于 2025-10-07) |
| labels | RFC |
| 热度 | 👍 13 · 💬 17 |
| 分类 | KV Cache / Memory |

## 一句话结论
提出 V1 中把设备 KV cache 卸载到 CPU:每步末生成 swap plan 交给 model runner 执行,BlockPool 分层重构并保持接口可扩展。

## 提案动机
当传输开销小于重计算时,把设备 KV cache 卸载到 CPU 可节省宝贵的 GPU 算力,长多轮对话等场景尤其有用;Nvidia C2C 等硬件改进大幅加速 CPU-GPU 通信,使卸载更有吸引力。

## 提出的方案
- 初始聚焦 CPU 卸载;接口与实现保持可扩展,未来可支持磁盘/远程存储等介质(本期不做)
- 采用 lazy swap-in/swap-out:每个调度步由 KV cache manager 累积各请求的换入/换出决策,步末生成 swap plan,作为 scheduler output 的一部分,由 model runner 在 forward 前执行;eager eviction、prefetching、逐层传输等优化可后续独立叠加
- KV cache manager API 微调:`get_computed_blocks` 额外返回 CPU 上已缓存的块;`allocate_slots` 为待 swap-in 的 CPU 块分配设备块;新增 `end_schedule_step` 钩子把完整 swap plan 存入 scheduler output
- BlockPool 重构为抽象基类,派生 `GpuBlockPool` 与新的 `CpuBlockPool`;新增 `_maybe_shelter_evicted_block` 供低层池接收上层淘汰的块
- 配置:复用 V1 中未使用的 `--swap-space` 控制 CPU cache 块数;当前 4GB 默认值需重新评估
- 性能:初始版用 async transfer(pinned memory、MemcpyAsync 和/或独立 stream)隐藏传输延迟;CPU 淘汰策略先用 round-robin,再加 LRU

## 优化点与收益
正文未给出量化数字。定性收益:在传输开销低于重计算时节省 GPU cycles,利用 CPU 的大容量存放 KV cache。

## 现状与落地
已关闭。正文计划四步(重构 BlockPool → 新增 CpuBlockPool 与 swap plan 逻辑 → 消费并执行 swap plan → benchmark 与低垂果实优化),并注明初始原型在 #13377;落地情况需查关联 PR。

## 关联
- 初始原型:#13377
- Nvidia C2C:https://www.nvidia.com/en-us/data-center/nvlink-c2c/

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
