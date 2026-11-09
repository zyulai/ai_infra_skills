# PR #23745: [Feat][EPLB] A novel static EPLB placement strategy for MoE models.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cboss6 |
| 合并 | 2025-09-16 |
| 分类 | MoE (tier: major) |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, gpt-oss |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/23745 |

## 一句话结论

为分组 top-k MoE 新增 round_robin(Zigzag)静态专家放置,DeepSeek-R1 单节点 h20x8 在线服务 QPM 提升约 8%。

## 动机

对 DeepSeek 这类多 expert group 的 MoE 模型,heatmap 分析显示同一 group 内的专家在实际负载中倾向被一起选中(grouped top-k 路由在组内做决策)。默认 linear 放置把连续专家块放在同一 rank,会使个别 rank 反复承接整组热点专家,造成 EP 组间负载不均。把同组专家错开分布到不同设备(Zigzag/round-robin)可以摊平负载;这是纯静态放置,不需要 EPLB 的动态重平衡与冗余专家。

## 优化点

新增专家放置策略 `ExpertPlacementStrategy = Literal["linear", "round_robin"]`:round_robin 下全局专家按 `arange(ep_rank, global_num_experts, ep_size)` 交错分配给各 EP rank(rank 0 拿 0,2,4...;rank 1 拿 1,3,5...),替代 linear 的连续块分配;`determine_expert_map` 按策略生 global→local 映射,支持专家数不整除 EP size 的情形(有余数的 rank 多拿一个)。入口为 `ParallelConfig.expert_placement_strategy` 与 CLI `--expert-placement-strategy`。适用性门控在 FusedMoE 层初始化时检查:round_robin 仅支持 `num_expert_group > 1` 且无冗余专家且未启用 EPLB 的模型(带冗余专家或无分组时不适用,不满足则打 warning 回退 linear)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/layer.py: determine_expert_map 增加 round_robin 分支与适用性检查/回退
vllm/config/parallel.py: ExpertPlacementStrategy 类型与 ParallelConfig.expert_placement_strategy
vllm/engine/arg_utils.py: --expert-placement-strategy CLI 参数
tests/distributed/test_expert_placement.py: 新增 round_robin 模式验证测试(含不整除与边界情形)
```

## 性能声明

在线服务验证(vllm/vllm-openai:v0.10.1.1,单节点 H20 x 8,TP8 + enable_expert_parallel):DeepSeek-R1 上 Zigzag 较默认配置 QPM(Queries Per Minute)提升约 8%。吞吐/延迟明细(DeepSeek-V2-Chat-0628,bench_serving random,input 1024 / output 128(正文表格行写 512,命令为 128),request-rate 8、max-concurrency 8、num-prompts 32)以图片形式给出,正文结论:仅开 expert parallelism 时 Zigzag 提升吞吐与端到端延迟约 3%。精度对照:Deepseek-v2-chat-0628 Aime24 13.33%→10.00%、Gpqa 41.91%→45.96%、Math500 72.20%→71.20%;Deepseek-R1-0528 Aime24 79.80%→80.00%、Gpqa 71.50%→71.21%、Math500 97.30%→95.20%;作者结论为对精度无影响(v2 在所选数据集上本身表现不佳,仅作对照)。

## 验证与回归风险

新增 `tests/distributed/test_expert_placement.py`:round_robin 在 world_size 2/4、专家数整除与不整除(如 9/2、17/2、18/4、33/4)下的 expert map 逐项断言;ep_size=1 返回 None map、ep_size=0 抛 AssertionError 的边界测试。风险面:round_robin 改变所有启用 EP 的 MoE 层的 global→local 专家映射,权重加载与路由都必须经过 expert_map(代码路径未变,但任何绕过 expert_map 的自定义实现会错位);与 EPLB 冗余专家互斥(未满足条件自动回退 linear 并 warning);QPM/吞吐数字来自作者在线基准,未附原始表(图片)。

## 关联

同区域:EPLB 动态专家负载均衡(eplb_config、enable_eplb、num_redundant_experts);本策略是 EPLB 之外的静态替代,作者注明 round_robin + EPLB 组合留待后续(TODO 注释)。pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
