# PR #47631: [Perf] Minimax M3 - Support cross-layer allreduce-norm fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-07-08 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | ready, v1, deepseek |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 7 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/47631 |

## 一句话结论

为 Minimax M3 打通跨层 allreduce-norm 融合,TP2 下总吞吐提升 2.2%。

## 动机

TP 切分下,MoE/FFN 层输出需要做跨卡 allreduce,紧随其后的又是下一层的 input RMSNorm,两个访存密集操作串行执行、重复读写 hidden states。DeepSeek-V3.2 路径已有 skip_final_all_reduce 机制把 MoE allreduce 融合进下一层 norm,但它是通过直接改写 `moe_config.skip_final_all_reduce` 实现的,且 Minimax M3(含 dense 层与 Gemma-style norm)没有对应支持。本 PR 把该机制泛化并接入 M3。

## 优化点

核心机制:让 FFN/MoE 层输出保持 un-reduced(`reduce_results=False`),把跨 rank allreduce 推迟到下一层的 input_layernorm(或最终 norm)中,经 `fused_allreduce_rms_norm`/`fused_allreduce_gemma_rms_norm` 一次完成。实现上分三层:(1) `FusedMoE` 新增 `reduce_results` 参数,在并行配置解析阶段统一推导 `skip_final_all_reduce`(与 all2all kernel、sequence parallel、zero expert 互斥);(2) MoE runner 在 final reduce 处跳过 allreduce 并用 assert 保证 fused output 未被提前 reduce;(3) M3 模型在 `MiniMaxM3Model.__init__` 里逐层扫描 FFN 是否延迟了 allreduce,据此设置各层的 `fuse_input_allreduce` 与 `fuse_final_norm_allreduce`,dense 层同样纳入。MTP block 与 PP>1 场景因 residual 直接相加/跨 stage 传隐藏态,强制走正常 reduce。

## 关键改动

```text
vllm/models/minimax_m3/nvidia/model.py: M3 各层 reduce_results=False、逐层配置 fuse_input_allreduce、最终 norm 走 fused_allreduce_gemma_rms_norm
vllm/model_executor/layers/fused_moe/layer.py: FusedMoE 新增 reduce_results 参数并推导 skip_final_all_reduce
vllm/model_executor/layers/fused_moe/runner/moe_runner.py: final reduce 阶段跳过 allreduce + 一致性 assert
vllm/models/deepseek_v32/nvidia/model.py: DSv3.2 从直接改 moe_config 迁移到 reduce_results=False 机制
vllm/models/deepseek_v32/nvidia/mtp.py: 注释同步(mtp_block 输出需手动 allreduce)
```

## 性能声明

TP2、InfX benchmark_serving:nvidia/MiniMax-M3-NVFP4,kv-cache-dtype fp8,random 数据集 input 8192 / output 1024,320 prompts、并发 32、request-rate inf、ignore-eos。Total Token throughput 13406.93 tok/s -> 13709.57 tok/s(**2.2% speedup**);Output token throughput 1499.27 -> 1533.11 tok/s,Mean TPOT 20.31 -> 19.87 ms。

## 验证与回归风险

PR 附带 GSM8k 评测(TP/TEP/DEP),5-shot flexible-extract exact_match 0.9098、strict-match 0.9090,无精度回退。风险点:allreduce 推迟逻辑与 MoE 并行模式(all2all kernel、sequence parallel、zero-expert)、PP、MTP 交互,`skip_final_all_reduce` 若与 pre-reduced 输出共存会触发 assert;Gemma-style norm 走新增的 fused 路径,需覆盖 final norm 的正确性。另顺手将 MLA prefill backend 的日志改为 `logger.info_once` 减少重复输出。

## 关联

同区域:diff 显示本 PR 把 DeepSeek-V3.2 已有的 skip_final_all_reduce 机制泛化(#62377 涉及 DSv3.2 sparse MLA CUDA graph);vLLM allreduce+norm 融合方向另有 #20691(SM100 FlashInfer fused allreduce)、#28841(GPT-OSS AR-norm fusion 修复)。
