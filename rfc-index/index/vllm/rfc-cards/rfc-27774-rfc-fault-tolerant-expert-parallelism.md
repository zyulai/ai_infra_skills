# RFC #27774: [RFC]:  Fault-Tolerant Expert Parallelism

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27774](https://github.com/vllm-project/vllm/issues/27774) |
| 作者 | tzulingk |
| 创建日期 | 2025-10-29 |
| 状态 | ⚪ closed(关闭于 2026-03-14) |
| labels | RFC, stale |
| 热度 | 👍 2 · 💬 11 |
| 分类 | MoE |

## 一句话结论
提议为 EPLB 增加专家级容错:检测-规避-重均衡-重启渐进恢复,服务不中断地隔离并恢复故障专家。

## 提案动机
现有 EP 实现假设所有专家在部署期间始终健康;生产环境运行 DeepSeek-V3 这类 256+ 专家、跨多 GPU 的模型时,个体专家故障不可避免:硬件退化(GPU 局部计算单元失效而其余正常)、内存错误(ECC 影响单个专家权重)、数值不稳定(特定专家因数据模式产出 NaN/Inf)、局部热点降频。无容错时,路由到故障专家的请求失败、流量突变引发级联负载不均、服务降级需人工干预。个体专家失败天然涵盖 GPU 级失败:某 GPU 全部专家失败时,重均衡算法会给它们分配禁绝权重,等效隔离整卡。

## 提出的方案
核心思路:容错即负载均衡——把健康监测作为 EPLB 的一等能力(扩展 `EPLBConfig`:health_check_enabled、health_latency_threshold=3.0、health_penalty_factor=10.0),渐进式恢复:检测→规避→重均衡→重启。
- Phase 0 健康监测与检测:低开销跟踪 per-expert 延迟;因 MoE 专家跑在融合并行内核(pplx、DeepEP 等)中无法直接测单专家,Phase 0 用 CUDA events 测整层前向耗时,按 router topk 把层延迟归属到该层激活专家、未激活记 0,在滑动窗口内对比各专家历史均值;延迟超过中位数 3×(可配)即判不健康,检测窗口约 100-1000 次前向(`EPLBConfig.window_size`)
- Phase 1 自动负载重分布:重均衡时给不健康专家 10×(可配)权重惩罚,流量自然转移到健康专家;无需新代码,仅参数传入
- Phase 2 专家重启:经弹性伸缩 scale-down 移除故障专家、scale-up 生成替换,再由 EPLB 重均衡(#27908,利用 #20323)
- 关键前提:EP 通信内核须内建容错——all_reduce 遇无响应 rank 不得永久挂起,应自动剔除失败 rank 并以健康 rank 集继续。内核支持矩阵(截至 2025-11-12):Mooncake EP(activeRanks masking)支持;DeepEP(mask_buffer_ptr masking)支持,但需扩展 buffer 初始化以从 Python 侧管理;pplx-kernels 不支持、会永久挂起
- 实施挑战:双层超时检测(GPU 内核层 + 编排层)需协同、DeepEP mask buffer 的分配/共享/同步、不同 rank 检测同一失败的时序不一致(可用多 rank 共识缓解)、部分失败下调用 `rearrange()` 的语义

## 优化点与收益
正文未给出量化收益数字(3× 阈值与 10× 惩罚为可配置参数而非实测结果)。定性优势:简单集成、永不停止服务只优雅降级、自动恢复、低开销;代价:检测需约 `window_size` 次前向的延迟、Phase 2 重启前存在容量损失。

## 现状与落地
已关闭(2026-03-14,label 含 stale)。正文未记录最终落地情况;Phase 2 指向关联 issue #27908,进展需查该 issue 及关联 PR。

## 关联
- Phase 2 依赖:https://github.com/vllm-project/vllm/issues/27908
- 弹性伸缩基础设施:https://github.com/vllm-project/vllm/issues/20323
- Mooncake EP:https://github.com/kvcache-ai/Mooncake
- DeepEP:https://github.com/deepseek-ai/DeepEP
- pplx-kernels:https://github.com/perplexityai/pplx-kernels

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
