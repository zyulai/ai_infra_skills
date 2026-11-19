# RFC #67874: [RFC][HiSparse] On-Demand KV Gather and Layerwise Two-Slot Reuse for Sparse-MLA (DSA) Prefill

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | ZD-euler |
| 创建日期 | 2026-11-16 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 主题分类 | Attention / Kernel |
| 链接 | [#67874](https://github.com/vllm-project/vllm/issues/67874) |

## 一句话结论
在 HiSparse 的 CPU offload 框架上补齐 sparse-MLA(DSA)prefill 的 KV 加载:短追加请求按 top-k 行并集做按需 gather,大 chunk 请求用双物理 GPU slot 跨层复用并让整页预取与计算重叠,同一 batch 内再按请求混合 FULL/DEMAND 两条加载路径。

## 提案动机
HiSparse 已把大块 sparse-MLA(DSA)主 KV offload 到 CPU 内存、轻量 indexer KV 保留在 GPU,decode 按需从 top-k 索引解析 KV;但其设计文档指出 prefill 与 prefill/decode 混合 batch 仍保留逐层(per-layer)staging。本提案补全 prefill 的数据加载与缓冲。长上下文 prefill 请求分两类:短追加请求(多轮 agent、编码负载常见)命中长 CPU 缓存前缀、新 token 很少,逐层计算很短,H2D 历史加载主导延迟;大 chunk 请求(chunked prefill 后段或前缀命中后的长追加)逐层计算窗口长,且并发与长序列给 GPU 显存容量带来压力,调度器常把两类请求合进同一 batch。当前 prefill 路径构建全量历史 staging 布局并从 CPU 读取所有非驻留历史行;TP 下 MLA 主 KV 跨 rank 复制,每个 TP rank 独立读取共享主机池的全量历史,主机读带宽压力被放大 TP 倍。在真实 agent 轨迹与仓库代码语料上实测:GLM-5.2(K=2048)下向 512K 历史追加 128 token,每层 top-k 并集平均仅覆盖历史的 1.2%–2.4%(外推到 1M 约 1%),即全量 staging 的主机读超过 97% 是无效读。

## 方案
保持现有调度、块分配(schedule / allocate_slots)与下游 sparse-attention 调用契约不变,分三阶段演进 prefill 路径:

- **PR1(top-k 过滤的按需 gather)**:用当前 batch 的 staging block table 把 prefill top-k 逻辑索引重映射到目标 staging 行;期间在 GPU 上、无主机同步地记录所有 prefill query 选中行的并集 mask,与现有 host-miss mask 求交后,只从 pinned 主机内存 gather 被选中的缺失行(GPU 驻留行也仅在被选中时拷贝);下游 sparse attention 直接消费重映射后的行索引。staging 保留全量布局、新生成 KV 写入偏移不变,唯一新增开销是在 GPU 上构建并集 mask;先支持 colocated 部署。
- **前置 NIXL PR**:扩展现有 NIXL CPU KV 支持,增加通用 producer 侧混合内存(DRAM + VRAM)导出,让 HiSparse(含 PR1)可在 disaggregated P-node 上运行,为 PR2/PR3 打通 P/D 交接。
- **PR2(逐层双 slot 复用 + 整页预取重叠)**:全量主 KV 历史逐层留在 CPU host pool;GPU 上 `num_blocks` 按(全层 indexer + 2 个主 KV slot)划分,经 `KVCacheTensor` 把奇偶层别名到两个 `num_blocks` 大小的物理 slot(Slot 0 / Slot 1),直接复用原生 `block_table` / `slot_mapping`。层 L 计算时后台预取流把层 L+1 历史块整页(FULL)载入另一 slot,slot 在此前读取与 CPU 异步回写都完成后才被覆盖;先面向 disaggregated P-node。设计承接 #48203 的逐层复用(其提出 2–4 个 buffer),计划与 @zhangsicheng5 及 #48203 团队协作对齐缓存分配与逐层加载接口。
- **PR3(同 batch 混合 FULL/DEMAND)**:沿用 PR2 的两个 slot,worker 侧按请求拆分加载模式:计算量大或并集宽的请求走 FULL 整页预取;长历史短追加请求走 DEMAND——history-only 块跳过整页预取,当前层 top-k 之后由 GPU kernel 只把选中的缺失行 gather 到对应块内偏移,其余块直接用 FULL 不重新分配。层 L+1 的 FULL 预取在层 L 的按需 H2D 完成后才开始,与层 L 的 sparse attention / FFN 及层 L+1 的 QKV / indexer top-k 重叠;两路填充完层 L 历史 KV 后整个 batch 合并到一次 sparse attention 调用执行。

## 优化点与收益
- PR1 消除短追加场景的无效主机读:补充材料实测,GLM-5.2(K=2048)、Q=128 时每层每请求并集稳定在 6.6K–8.7K 行,覆盖 H=64K/128K/256K/512K 历史的 12.6%/6.6%/2.9%/1.25%(约 3.2×–4.2× K);Q=1024 时 H=64K 覆盖升至 32%–45%,H=512K 为 4.0%–5.7%(约 21K–30K 行)。长上下文仓库代码(194 个拼接源文件)上,Q=128→1024、H=64K→512K 的并集覆盖介于 8.6%–36.5%(如 64K+128 为 8,838 行/13.5%,512K+1024 为 45,218 行/8.6%);256K+Q=64 平均并集 8,378 行(3.2%)。逐层并集覆盖差异约 4×(128K+1024 个别层达 40.4%,64K+1024 达 53.3%),与 PIVOT 论文(arXiv 2607.24593)在 DeepSeek-V3.2(128K RULER-QA)上报告的 84%–90% query 间共享位置一致——小 Q 时并集随 H 增长极慢(适合按需 gather),大 Q 或中等 H 时并集迅速变宽(适合整页预取),据此按 H 与 Q 路由请求。
- PR2 显存收益:P-node 免除逐层 GPU 驻留主 KV 与本地 decode 热缓存;GLM-5.3 FP8 KV(78 层主 KV + 21 个共享 indexer)下,双 slot 复用把物理主 KV 占用缩到标准全层 GPU P-node 的 2/78,总 GPU KV 内存(含 indexer)节省约 92.4%,同 HBM 预算下可分配 `num_blocks` 扩大约 13.2×。
- 正文坦承尚未完成端到端 GPU benchmark;后续将按阶段验证:PR1 测 GPU top-k 并集比例、验证输出等价并对比主机读字节与 gather 带宽;PR2 查 GPU 时间线验证跨层传输/计算重叠(含多 GPU TP 并发预取下的主机带宽上限)、固定 HBM 下最大并发、同并发 TTFT/吞吐对比及 P/D 交接完整性;PR3 在同两 slot 上对比全 FULL 与全 DEMAND 基线。

## 现状落地
- 状态 open(2026-11-16 创建,👍 0 · 💬 0),feedback period 一周;CC @MatthewBonanni @LucasWilkinson @NickLucche @ivanium @orozery @zhangsicheng5 @LCAIZJ。
- 正文未给出已合并实现或已提交的关联 PR;PR1、NIXL 前置 PR、PR2、PR3 均为规划。落地思路:先基于 HiSparse 现有 KVConnector layerwise load/save hooks 把双 slot 逐层复用落地为 P-node prefill 驻留模式,再与 #48203 作者及维护者评审哪些抽象应提升为通用 KVConnector 接口。
- 关联:#33398、#48203(逐层复用设计);HiSparse 设计文档 docs/design/hisparse.md;源码基线 commit 55aa0cab。

---
> 本卡由 RFC 正文自动素材 + LLM 撰写,未经人工核实。
