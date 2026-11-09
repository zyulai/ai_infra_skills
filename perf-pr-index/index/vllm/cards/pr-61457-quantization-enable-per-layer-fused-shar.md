# PR #61457: [Quantization] Enable per-layer fused shared expert selection

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-10-28 |
| 分类 | Quantization（tier: major） |
| Label | ready, qwen, deepseek, cpu, quantization, DSv4, glm, minimax, DSv4.1 |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；pack 中 diff 截断，省略 36 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/61457 |

## 一句话结论

FSE（AITER fused shared experts）从模型级 all-or-nothing 改为逐层选择：敏感层保留高精度 shared expert、其余层融合，混合策略保住生成质量并扩大 FSE 适用面。

## 动机

当前 vLLM 在任一层无法启用 AITER fused shared experts（FSE）时拒绝整个模型，多处逻辑假设 all-or-nothing。动机是：部分层对 MXFP4 shared expert 量化不敏感、可以融合，另一部分层敏感、应保留更高精度（如 MXFP8/BF16）的独立 shared expert——一刀切导致要么全部不融合（放弃性能），要么全部量化（伤精度/生成长度）。

## 优化点

- 通过既有 `resolve_layer_fused_shared_expert` API 实现逐层 FSE，覆盖已迁移到 layer-scoped checkpoint loading 的模型；`get_fused_shared_expert_layers()` 收集融合层索引集合，替代被删除的 `is_model_fused_shared_expert_compatible()`。
- checkpoint 权重重定向只对分配了 fused shared-expert slot 的层生效：融合层的 shared expert 权重改写入 routed experts 的附加 slot，未融合层保留独立 shared expert（可与 MXFP4 routed experts 并存 MXFP8/BF16 shared expert）。
- 显式拒绝混合 FSE 与 EPLB、expert parallelism、`nixl_ep` all2all backend（三者假设各层 expert geometry 一致；`nixl_ep` 的进程级 buffer 按"首个 MoE 层的 num_experts_per_rank"初始化，混合模型会静默错位）。
- 迁移 MiniMax M3、DeepSeek V4/V2/V3.2、GLM5Next、AXK1、GLM4 Lite 及相应 MTP loader 到该 API；EP/DP eligibility 用显式 `disable_under_ep` 调用点参数替代 model-type 查表。`vllm serve` 启动时输出去重后的逐层策略（哪些层 FSE、哪些层保留独立 shared expert、每个 skip reason 一行）。
- 兼容性变化（breaking）：AITER FSE 用于 MiniMax M3/DeepSeek V4 现在同样要求 `VLLM_ROCM_USE_AITER=1`；`maybe_fuse_shared_experts()` 需要 `fused_shared_expert_layers` 集合；`FusedMoEConfig.aiter_fmoe_shared_expert_enabled` 改为 `is_fused_shared_expert_requested`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/utils.py: resolve_layer_fused_shared_expert / get_fused_shared_expert_layers 逐层策略
vllm/model_executor/models/utils.py: maybe_fuse_shared_experts 按 per-layer 集合重定向 checkpoint 权重
vllm/models/deepseek_v4/amd/model.py 等 13+ 架构 loader: 迁移到逐层 resolver
tests/model_executor/layers/test_fused_shared_expert.py: 13 架构 init/加载/策略断言
```

## 性能声明

PR 无 kernel 吞吐提升数字；"60% faster" 指测试自身（`test_models_fse_init` 用 `single_rank_parallel_state` 替代 `dist_init`，该测试快约 60%）。端到端价值体现在精度/生成长度混合策略对比（gsm8k_platinum，1209 样本 × 5 repeats，MI350，amd/Qwen3.5-35B-A3B-MXFP4，TP=2）：

- shared experts 全不量化（baseline）：strict/flexible 0.9843/0.9788，mean gen tokens 3543；
- 仅 layers 10–20 量化（混合 FSE）：0.9866/0.9798，3568——与 baseline 持平略优，且这些层可融合；
- 全部量化（全层 FSE）：0.9835/0.9763，3873——生成长度劣化约 9%（median 3327→3767，p90 5820→6112）；混合策略保持长度（median 3327→3385，p90 5820→5789）。

## 验证与回归风险

- 单测（MI350）：main 105 passed → 本 PR 166 passed / 3 skipped；`test_models_fse_init` 扩到 13 架构（含 MiniMax M3、两层 DeepSeek V4、Qwen4 Exp），覆盖 MXFP4-only 与混合 MXFP4/MXFP8 shared experts、MTP loading、EPLB/EP/`nixl_ep` 拒绝路径；非 ROCm 上断言 unfused 结果而非 skip；新增 ModuleList（0-based）与 ModuleDict（绝对索引）索引空间、DeepSeek V4 mapper fallback 用例。
- 端到端：Quark 离线量化 DeepSeek-V2-Lite（layers 5-7 shared experts 留 BF16）与 `amd/Qwen3.5-35B-A3B-MXFP4` 在线 `--quantization-config` 定向量化两条路径的启动日志均确认逐层策略正确解析。
- 风险（PR 自述限制）：混合 FSE 与 EPLB/EP 不支持；同一 fused GEMM 内不允许异构精度；fused 层须共享 routing-buffer geometry；Mori 不支持；forward 执行、graph replay 与性能未验证（PR limitations 明示，尽管 gsm8k_platinum 评测已跑）；NVIDIA 模型路径也被改动（虽然 FSE 在该平台不支持）；三个 follow-up 未完成（FHMOE 逐层策略迁移、DSV4 MTP/V4.1 loader 消费 `fused_shared_expert_layers`——目前构造时 raise NotImplementedError、移除 NVIDIA 侧死代码）。

## 关联

- 相关 PR：#57916（ModelOpt MXFP8 兼容，不同范围）；#62092（AITER 默认开启，使本 PR 的 `VLLM_ROCM_USE_AITER=1` 要求成为过渡）。
- 同分类：本批 PR #51473、#45739、#28892、#32520、#28603。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
