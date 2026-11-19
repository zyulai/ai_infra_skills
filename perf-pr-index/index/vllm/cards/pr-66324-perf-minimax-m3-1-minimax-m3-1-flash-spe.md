# PR #66324: [Perf][MiniMax-M3.1] MiniMax-M3.1-Flash Specialized Decode: CuTe DSL Optimized GB300 (up to 1.45x)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xinli-sw |
| 合并 | 2026-11-16 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, minimax, agents |
| Diff | pack 标注 +None/-None, None files；pack 末尾说明「38 file(s) dropped from diff」（正文自述约 25.4k 新增行，多为 kernel 代码） |
| 链接 | https://github.com/vllm-project/vllm/pull/66324 |

## 一句话结论

为 MiniMax-M3.1-Flash 在 GB300（TP4）加入 NVIDIA 开发的专用小批量 decode 执行器（CuTe DSL kernel 以 PDL 链接），AgentX agentic trace p90 交互性最高 1.45x（c=1：550 → 799 tok/s/user），decode 步 c=1 提速 32–33%。

## 动机

agentic 与交互式 serving 的 decode batch 很小（c=1–8），通用前向在此区间受 launch 与边界开销主导。沿用 #61867 与 #61868 的集成模式（模型自持 decode executor 接管 FULL-graph decode 步），为 MiniMax-M3.1-Flash（模型本体来自 #64111）提供专用 decode 路径：GB300 单 tray TP4（GB200/B200 亦可）、无投机或 DSpark 最多 7 个 draft tokens、最多 64 行（8 requests × (K+1)）的 FULL-graph decode 步；prefill、混合 batch 与更大 batch 回退 stock 前向。opt-in 开关 `--additional-config '{"decode_backend": "latency_specialized"}'`；默认 `auto` 下与 upstream main 完全一致。所有 kernel 为 CuTe DSL（`nvidia-cutlass-dsl`，已是 vLLM 依赖），warm-up 时按当前设备 JIT 编译，不改 setup.py/CMake。

## 优化点

- **每层融合链（sparse MoE 层 3–59，PDL 链接）**：`qkv_front -> index_score -> page_topk -> sparse_decode -> o_proj(+router CTAs) -> post(AR+norm+NVFP4+top-4 routes) -> fused NVFP4 MoE -> (push) 下一层 qkv_front`。
- **QKV front**：一次 launch 完成上个子层 TP all-reduce 落地（peer exchange）、residual add、Gemma RMSNorm、MXFP8、QKV/index-QK GEMV（tcgen05）、per-head QK norm + RoPE、fp8 KV 与 index-K 插入。
- **Index score / page top-k / sparse decode**：上游 CuTe DSL scorer 换 split；exact page top-k 直接写物理 page id（>4096 有效页按 1024 页 chunk CTA 切分，最后到达的 chunk 合并，表逐位一致）；fp8 sparse decode 输出 o_proj 输入的 MXFP8。
- **Post + MoE**：o_proj GEMV + router CTAs，post kernel 一次完成 all-reduce + residual + norm + FC1 输入 NVFP4 量化 + top-4 路由；fused NVFP4 MoE engine（routed + shared experts）原地读 vLLM 的 TRT-LLM-gen expert buffers 并把输出 push 给下一层 front。fused post+MoE engine 止于 8 行，故完整 kernel 链仅在 c=1/K=7 跑满；9–16 行用 wide GEMV o_proj，17–64 行 front/o_proj 均用 wide GEMV（见 rows 分发表）。
- **DSpark drafter**（保留 stock 模型）：Markov 链每位置单 kernel 采样、norms 与下一次 MXFP8 量化融合（`draft_norm.py`，all-reduce 融合实测慢于 FlashInfer MNNVL AR 而未做）、经 CUDA IPC peer memory 读取双 LM head logits（`peer_logits.py`）替代 NCCL all-gather。
- **Rank lockstep 与 fail-closed**：`unsupported()` 每 rank 运行并 MIN all-reduce 合并；kernel build 分阶段 collective 同步，任一 rank 失败的 stage 全 rank 回退 stock。
- **MoE finalize defer 上限**：开关打开时 `fused_finalize_max_tokens` 把 stock 的 32 行上限提到 MAX_ROWS=64。

## 关键改动

```text
vllm/models/minimax_m3/nvidia/specialized_decode/（新包）:
  __init__.py: SUPPORTED_TP_SIZES=(4,), MAX_SPECULATIVE_TOKENS=7, MAX_REQUESTS=8, MAX_ROWS=64,
    PAGE_SIZE=128, MAX_PAGES=8192; DecodeConfig; unsupported()（逐项列出缺失需求）; specialized_decode_config()
    （TP 组 MIN all_reduce 后不一致则全体 raise）
  stages.py / stock.py / kernel_stages.py / kernels/（front_kernel.py, page_topk_kernel.py,
    sparse_decode_kernel.py, post/, o_proj/, moe/, wide.py）
  drafter.py / draft_norm.py / peer_logits.py: DSpark drafter kernel（sample_sequential、FusedGemmaNormQuant、IPC logits）
  runtime/: decode.py（SpecializedDecode.plan()/run(), DecodeStep）、model_state.py、warmup.py
  AGENTS.md: 结构、dispatch/fallback 混合规则表、加 kernel stage 步骤、serving 命令
vllm/models/minimax_m3/nvidia/model.py:
  + self.specialized_decode_config = specialized_decode_config(vllm_config)
  + def build_specialized_decode(self): ... SpecializedDecode.for_target(self.model, ...)
  + def forward(...): step = decode.plan(positions, intermediate_tensors); ... decode_step=step
  + MiniMaxM3Model.forward(..., decode_step): if decode_step is not None: return decode_step.run(hidden_states, positions)
  + _init_fused_moe_finalize 用 fused_finalize_max_tokens 替代 FUSED_MOE_FINALIZE_MAX_TOKENS
vllm/models/minimax_m3/common/dspark.py: 开关打开时 attach_drafter(self, vllm_config)
```

## 性能声明

（所有运行：MiniMax-M3.1-Flash，GB300 单 tray TP4，fp8 KV 与 fp8 index-K 缓存，checkpoint 的 MXFP8/NVFP4 权重，Model Runner V2 + async scheduling，FULL CUDA graphs decode，DSpark K=7；两臂以 synthetic rejection 强制 acceptance length 4.04 以核对同量 tokens/步。）

- **AgentX**（InferenceX agentic coding traces，固定并发闭环，每 cell 20 分钟；p90 交互性 = 1/ITL，在更早分支版本上测于 main@2b009fc6d）：c=1 550 → 799 tok/s/user（**1.45x**），c=4 450.2 → 554.1（1.23x），c=8 358.7 → 457.5（1.28x）；对照 InferenceX GB300 vLLM 575/470/350、其 best (SGLang) 658/485/417（为参考而非受控对比）。
- **Decode step**（本 head，main@463e23b1d 真实权重，稳态 step p50 over 256 decode tokens，双启动镜像取均值；off 列约 ±5% run-to-run 噪声）：8K/c1 6.99 → 4.76 ms（−32%，1.47x）；8K/c2 8.00 → 6.54（−18%，1.22x）；8K/c4 9.79 → 8.22（−16%，1.19x）；8K/c8 12.31 → 10.80（−12%，1.14x）；128K/c1 6.90 → 4.61（−33%，1.50x）；128K/c2 7.64 → 6.41（−16%）；128K/c4 9.99 → 8.57（−14%）；128K/c8 12.70 → 11.80（−7%，1.08x）。
- **长上下文多请求步**（旁挂 524K-token 请求，step p50 ms，无/有 split-row page top-k 与 pre-wait sparse decode）：524K+16K（16 行）8.36 → 8.04（省 0.32 ms）；524K+3×16K（32 行）10.65 → 9.97（省 0.68 ms）；524K+7×64K（64 行）12.46 → 12.04（省 0.42 ms）。
- **精度**：GSM8K 5-shot strict/flexible（c=1/c=8）、GPQA-Diamond mean@12、AIME26 mean@16、RULER NIAH 32K/128K/256K 共 15 行全部在 |on−off| ≤ max(1 SE, 1 point) 内（如 GSM8K strict c=1 0.9629 → 0.9636；GPQA-Diamond 0.843 → 0.850）；DSpark acceptance length 7 行变化 −2.0% ～ +0.8%（bar ±3%）全部 pass。
- **已知口径差异**：CuTe DSL build 相对更早 build 平均步比 0.985（128K c=1 慢 4.0%：4.836 vs 4.652 ms，nsys 下 GPU 时间相同）；测量 build 带三个未随本 PR 落地的 Model Runner V2 opt-in（每步约 30–40 µs，约 c=1 步时的 0.7%）；本 head 的 GPU 复验 pending，所有数字在旧 base 上测得、正在本 head 重跑。

## 验证与回归风险

15 个 score 行与 7 个 acceptance-length 行全部 pass（对照同 build 开关关闭臂，TP4、fp8 KV/index、DSpark K=7 probabilistic drafting + lossless block verification、max-num-seqs 8、max-model-len 1048576）。Kernel 级测试（float64 参考、TP4 边界交换 graph replay、逐 stage 对照 stock、五层模型开关 A/B）曾跑于 4×GB300，但未随本 PR 提交（保持可 review），将另出 PR 加 CI 步骤；GPU 复验 pending。回归风险：约 25.4k 新增行，作者请求对 kernel 细致人工 review；仅限 SM100/SM103、TP4 全 world、Model Runner V2 + async scheduling、DSpark ≤7 tokens、fp8 KV/index（NVFP4/BF16 拒绝）、block size 128、≤8192 pages、FULL decode graphs、无 torch.compile/sleep mode/LoRA/batch invariance/KV transfer（除 SimpleCPUOffloadConnector），不满足则全 rank 启动即 raise；fp8 KV 在 GB200 TP4 是 3.66M-token KV pool（NVFP4 为 6.55M），c=16 时会驱逐等待会话的缓存前缀（32.1M prompt tokens 重算 vs stock NVFP4 的 11.3M，中位数领先 448 vs 413 但 p90 落后 264 vs 292 tok/s/user）；attention metadata 仍每步 eager 构建；post kernel 共享内存上报为 0（coresidency 检查最多少计 304 B）；compute-sanitizer（memcheck/synccheck/racecheck）尚未运行。

## 关联

- 集成模式：#61867、#61868（同开关 decode_backend=latency_specialized 的前例）；模型：#64111。
- 上游变更涉及测量路径：#61813（MiniMax-M3.1 改读 Transformers MiniMaxM3VLConfig）、#64476（DSpark Markov 采样融合 Gumbel kernel，改变 off 臂 drafter kernel）；#65386（ROCm-only DSpark drafter monokernel，NVIDIA 上不运行）。
- 后续计划：#63406 的 decode capture-shape helpers（替换从 cudagraph_capture_sizes 推导 step rows）；NVFP4 KV/index-K 支持；TP8/16；fused post+MoE >8 行；attention-metadata CUDA graphs；Model Runner V2 opt-ins 单独 PR。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
