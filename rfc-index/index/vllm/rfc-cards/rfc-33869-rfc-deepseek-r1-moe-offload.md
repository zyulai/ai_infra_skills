# RFC #33869: [RFC]:DeepSeek-R1 Moe offload

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33869](https://github.com/vllm-project/vllm/issues/33869) |
| 作者 | wangyxbh |
| 创建日期 | 2026-02-05 |
| 状态 | ⚪ closed(关闭于 2026-07-28) |
| labels | RFC, stale |
| 热度 | 👍 15 · 💬 7 |
| 分类 | MoE |

## 一句话结论
提议 MoE 推理 CPU offload 模块:GPU 只缓存热点专家,冷专家驻留 CPU 以 AVX/AMX 内核计算,双模式流水重叠。

## 提案动机
DeepSeek 类大型 MoE 推理受 GPU 显存压力:专家权重占显存大头,而每 token 只激活少量专家。现有做法要么把全部专家权重放 GPU,要么朴素 CPU fallback、同步开销高。正文的关键观察:专家访问稀疏且有偏;CPU 内存容量足以容纳全量专家权重;现代 CPU(AVX512/AMX)经优化后可高效计算 MoE FFN;GPU 内核可借回调与 CPU 执行重叠。目标:降显存不牺牲吞吐、细粒度专家缓存、GPU 计算/CPU 计算/数据搬运三者重叠、与 vLLM 执行模型干净集成。

## 提出的方案(关联 PR #31938)
- 核心设计:GPU 不再存每层全部专家权重,只维护有限容量的热专家缓存;CPU 驻留全量专家(pinned memory),按实际 token 路由动态决定哪些专家复制上 GPU、哪些直接在 CPU 计算;每层维护 cache_map/miss_map/policy_sort 元数据
- 组件:Python Offload Manager(CpuOffloadInfer)编排全局;GPU Expert Cache;双缓冲 Miss Expert Buffer(全局仅两个缓冲,按 `layer_id % 2` 复用,配独立 CUDA stream/event 使相邻层拷贝与计算交错流水);AVX/AMX 优化的 CPU MoE 执行引擎;GPU↔CPU 回调式异步同步
- 执行模式一 DBO(Dual Batch Overlap):decode/小批量场景,CPU 计算、GPU 计算、专家拷贝置于不同线程,借 vLLM DBO 调度重叠以降低 decode 延迟
- 执行模式二 Prefetch:较大 prefill 批,按 token 数动态计算 `n_copy` 限制本 pass 最大拷贝数,拷入 Miss Expert Buffer 供 GPU 执行,其余专家由 CPU 计算并在输出阶段合并;当 `n_copy` 覆盖全部专家时自然退化为全 GPU 模式

## 优化点与收益
正文未给出量化数字(定性表述:显著降低 GPU 显存占用,提升数千专家级大 MoE 的可扩展性)。

## 现状与落地
已关闭(2026-07-28,label 含 stale)。正文未记录最终落地情况,需查关联 PR #31938。

## 关联
- PR #31938:https://github.com/vllm-project/vllm/pull/31938

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
