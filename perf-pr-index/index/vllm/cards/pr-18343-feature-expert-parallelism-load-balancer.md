# PR #18343: [Feature] Expert Parallelism Load Balancer (EPLB)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @abmfy |
| 合并 | 2025-06-26 |
| 分类 | MoE |
| Label | performance, ready, ci/build, v1, qwen |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/18343 |

## 一句话结论

为 EP 部署引入动态专家负载均衡(EPLB)与冗余专家副本,按滑动窗口负载周期性在 EP rank 间重排专家权重,缓解 aux-loss-free MoE 的热点专家。

## 动机

DeepSeek-V3/R1 等 auxiliary-loss-free MoE 模型在推理时无训练期负载均衡损失约束,专家负载天然不均,EP rank 间 token 分布失衡拖慢整体。解法有二:动态把专家在 rank/node 间重排;为热门专家建多份冗余副本分摊路由。

## 优化点

三块组成。状态与度量(`eplb_state.py`):`EplbState` 维护 physical↔logical 专家映射、`expert_load_pass` 每步 token 计数与 `expert_load_window` 滑动窗口,每 `eplb_step_interval` 步触发一次 `rearrange`(初始步设为 interval 的 3/4 以尽早首排);可选每步 all_gather 输出 balancedness(avg/max load,默认关闭因其有通信开销)。重排算法(`rebalance_algo.py`,改自 deepseek-ai/eplb):group→node→GPU 三层 hierarchical `balanced_packing`(贪心装箱)+ `replicate_experts`(贪心复制 `weight/logcnt` 最大者,最小化最大负载);group 数不整除 node 数时退化为 global 策略。权重迁移(`rebalance_execute.py`):逐层 `shuffle_layer`,本地可满足的副本直接 copy,跨 rank 用 `batch_isend_irecv` P2P 交换到临时 buffer;profile 模式用 dummy all_gather 预留通信 buffer 防止 profile_run OOM。模型侧通过 `MixtureOfExperts` 协议接入(deepseek_v2 暴露专家权重与映射,FusedMoE 支持冗余专家路由,`expert_params_mapping` 反映副本)。

## 关键改动

```text
vllm/distributed/eplb/eplb_state.py: EplbState(负载窗口度量 + 周期重排驱动)
vllm/distributed/eplb/rebalance_algo.py: hierarchical/global 重排算法(DeepSeek EPLB 适配)
vllm/distributed/eplb/rebalance_execute.py: 逐层专家权重 P2P 迁移
vllm/config.py: --enable-eplb / --num-redundant-experts / --eplb-window-size / --eplb-step-interval
```

## 性能声明

PR 未提供量化 benchmark 数字(无吞吐/时延对比);定位为负载均衡机制引入,收益取决于负载失衡程度。PR 说明监控日志本身有通信开销、可关闭。

## 验证与回归风险

`tests/models/test_initialization.py` 调整(DeepSeek 类小模型的专家数/grouped_topk 约束);pack 中 18 个文件被裁剪,新增测试不可见。风险面:仅 CUDA;冗余专家使 weight loading 显著复杂化(PR 明示需在模型类里做非平凡调整);rearrange 期间逐层顺序 shuffle、每层 `torch.cuda.synchronize()`,会造成推理停顿(PR 的长期 TODO 即"逐层分步 shuffle 以降低 ITL 影响"、并行化算法等);当前仅支持 DeepSeek-V2/V3/R1 + FP8。

## 关联

同区域:改自 deepseek-ai/eplb 上游实现;长期 TODO 指向 MTP 层支持、two-batch overlap(#18415)、Llama 4/Qwen3 等其他 MoE 模型与其他量化方法;EPLB 重排与后续 WideEP/MoE 通信优化 PR 配合使用。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
