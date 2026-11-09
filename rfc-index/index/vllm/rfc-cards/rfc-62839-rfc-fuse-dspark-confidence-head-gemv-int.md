# RFC #62839: [RFC]: Fuse DSpark confidence-head GEMV into a single TileLang kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62839](https://github.com/vllm-project/vllm/issues/62839) |
| 作者 | panpan0000 |
| 创建日期 | 2026-10-21 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |

## 一句话结论
把 DSpark 置信头(cat + cast + GEMV + sigmoid)融成单个 TileLang 内核:一次 CUDA launch、一个 CUDA-graph 节点、零 fp32 临时,且保持 FP32 权重语义。

## 提案动机
DSpark 自适应推测解码每个 draft step 跑一次小置信头,现状是 4 次 CUDA launch(`torch.cat`、`.float()` cast、cuBLAS GEMV、`.sigmoid()`)加一个 `(n, H + r)` fp32 临时,且绕过 CUDA-graph 内存复用;GEMV 本身极小(`H + r ≈ 7 200`),RTX 5090 上四次 launch 约 117 µs CPU 调度、约 14–45 MB 激活流量,而有效答案约 2 µs 长。DeepInfer(DeepSeek 自家引擎)已用 60 行 TileLang 内核 + PDL 出货等价融合,但把权重 cast 成 bf16;vLLM 权重保持 FP32(`params_dtype=torch.float32`),直接 drop-in 移植会静默改变 draft token 的接受结果。

## 提出的方案
- 移植 DeepInfer 融合但保持 FP32 权重:113 行 TileLang 内核(fp32 累加器、内核内 sigmoid),经 `VllmJitKernel` 包裹(引擎 JIT warmup 预编译)、`direct_register_custom_op` 注册为 `torch.ops.vllm.dspark_confidence_tilelang`,CUDA-graph 捕获为单节点;代码在 PR #62840。
- 严格 dispatch 门:SM90+ 平台、hidden/markov 为 bf16、权重 fp32、shape 仅 `(n_embed = 7168, markov_rank = 32)`(DeepSeek-V4-Flash)、`32 ≤ n ≤ 2048`、last-dim 连续等;任一不满足即回退既有 PyTorch 路径,无用户可见行为变化。
- 定位为 DSpark 小算子融合系列的首个移植(后续如 `dflash_input_store_tl`、`dspark_draft_embed_store_tl`),机制级收益可叠加。
- 回退可观测:wrapper 对每个唯一拒绝原因记一行 `logger.debug`(默认关闭)。
- 留给 reviewer 的开放问题:30-prompt greedy 行为一致性是否足以 gate 数值风险。

## 优化点与收益
条件:RTX 5090(SM120,CUDA 13.0)与 H200(SM90,CUDA 13.1)各一卡;`Qwen/Qwen3-4B-FP8` + `deepseek-ai/dspark_qwen3_4b_block7`,vLLM 0.32.0、TileLang 0.1.12。对 `torch.compile` 的中位内核延迟加速:`1.92×`(5090,36 形状)、`1.90×`(H200,20 形状);每 decode 步 CUDA launch `4 → 1`(−3)、CUDA-graph 节点 `4 → 1`、fp32 临时 `14 – 45 MB → 0 B`;真实 checkpoint greedy A/B `0/30 divergent outputs`(双 GPU 输出与 token 数逐位一致);TPOT 级增益 `< 0.05 %`,内核占总 kernel 时间 `0.03 %`(5090)/`0.05 %`(H200)。另观测 H200 4B 形状串行回归约 50%(fused 均值 4.95 s vs baseline 3.29 s),现已被 shape 门排除在可达路径之外。

## 现状与落地
仍在讨论/已有原型:open 状态,实现在 PR #62840(29/29 内核测试通过),反馈期为两周或直到 `vllm/model_executor/kernels/` 或 `vllm/models/deepseek_v4/` 至少一位 maintainer 回应;尚待 maintainer 反馈。文末注明 AI(Claude)协助起草内核/测试/基准,作者复审并跑测试。

## 关联
- [PR #62840(初始实现)](https://github.com/vllm-project/vllm/pull/62840)
- [DeepInfer `dflash.py:12-72`(同款融合,bf16 权重)](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py#L12-L72)
- [PR #41536(mHC post_pre 融合先例)](https://github.com/vllm-project/vllm/pull/41536)
- [PR #43905(mHC `kernels/` 目录布局)](https://github.com/vllm-project/vllm/pull/43905)
- [#47456(JIT warmup infra RFC)](https://github.com/vllm-project/vllm/issues/47456)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
