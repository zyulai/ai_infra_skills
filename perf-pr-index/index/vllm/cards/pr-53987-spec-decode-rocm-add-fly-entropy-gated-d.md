# PR #53987: [Spec Decode][ROCm] Add FLy: entropy-gated deferred verification for draft-model speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @eecspan |
| 合并 | 2026-10-19 |
| 分类 | Speculative Decoding |
| Label | documentation, rocm, speculative-decoding, ready, mrv2, verified |
| Diff | diffstat 缺失(pack 未提供,正文 diff 另有 2 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/53987 |

## 一句话结论

新增 FLy 有损投机验证策略:高熵位置的孤立 rejection 可被推迟,MRV2 上 AMD 1.055x-1.310x、NVIDIA 1.041x-1.273x 解码加速,无需额外 checkpoint 或 target forward。

## 动机

标准投机验证在第一个被拒绝的 draft token 处停止,后续已可接受的窗口全部作废。FLy(arXiv 2511.22972)允许在 target 熵足够高、且后续 W 个 draft token 全部通过原生验证时接受一个孤立 rejection,不增加任何 checkpoint 或 target forward。代价是不保持 target 分布、可能降低输出质量,因此为 opt-in,标准验证仍是默认。

## 优化点

FLy 只改验证判定,复用各采样模式的原生接受规则(greedy target 用 token 相等;greedy drafting + random target 用 target-only;概率 drafting 用 probability-ratio)。一个被拒绝位置满足以下条件才可推迟:draft token 未被 target 约束 mask、其 top-k target 熵达到阈值、且其后连续 W 个 draft token 全部原生通过;窗口内再出现 rejection 则推翻原判定;尾部不足 W 的窗口不推迟。熵按 `-sum(p*log(p))` 在最大的 `fly_entropy_top_k`(默认 3)个已处理 target 概率上计算,不重归一化(实现用 top-k 门控,而非论文的全词表归一化熵)。默认 `fly_window_size = min(6, num_speculative_tokens - 1)`,要求至少 2 个 speculative token。实现分两层:MRV1 用 pre-pass 覆写 target argmax/uniform sample,把 FLy 决策编码进现有接受 kernel 的输入;MRV2 在 rejection kernel 内追踪 pending rejection,窗口全部原生接受后才 commit,失败窗口在原拒绝位置恢复 argmax/概率归一化统计。Triton kernel(greedy/random 两个)与 `compute_fly_entropy` 在新增 `vllm/v1/spec_decode/fly.py`;FLy 专属工作仅在显式开启时运行。

## 关键改动

```text
vllm/v1/spec_decode/fly.py: 新增熵计算 + greedy/random 两个 Triton 覆写 kernel(188 行)
vllm/config/speculative.py: rejection_sample_method 增加 "fly";fly_window_size/threshold/top_k 配置与校验
vllm/v1/sample/rejection_sampler.py: MRV1 接入(pre-pass 覆写 argmax/uniform;CPU 不支持)
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py: MRV2 kernel 增加 pending_rejection/FLY_WINDOW_SIZE 逻辑
docs/features/speculative_decoding/fly.md: 新文档,标注 lossy warning
```

## 性能声明

MRV2(4 模型 DSpark/MTP,MI355X 与 B300,batch 1-64,window=2,threshold 0.3,top-k 3):48 个 model/platform/batch 点全部快于 Standard SD,AMD 1.055x-1.310x、NVIDIA 1.041x-1.273x(例:Qwen3.5-122B-A10B AMD batch 64 3,349.1 → 4,385.7 tok/s,1.310x;DeepSeek-V4-Flash AMD batch 1 206.9 → 228.2 tok/s,1.103x)。质量(GSM8K/MATH-500/AIME25/GPQA-Diamond 单次 pass@1):相对 Target 最低保留率 AMD 96.43%、NVIDIA 96.30%(均为 AIME25 一题之差)。MRV1(4 对 target/draft,10 benchmark,MI355X,torch.compile+HIP graph,温度 0):40 个配置全部更快,1.067x-1.938x,典型约 1.23x;与 DFlash 对比(Kimi-K2.5 1T,MI300X,8K in/1K out)吞吐 >1.09x,GSM8K 保留 99.7%。

## 验证与回归风险

新增 `test_rejection_sampler.py` 的 FLy 测试组(greedy rescue、window 边界等)与 chunking 测试的 FLy 字段初始化;配置层校验 window/熵参数并一次性输出 lossy 警告。风险:文档明确警示 FLy 破坏分布保持保证(质量逐模型/任务波动,如 Llama-3.1-405B GPQA-Diamond -2.70 pp、R1-Distill-70B AIME 2024 20/30 → 17/30);MRV1 路径 CPU 不支持;`use_heterogeneous_vocab=True` 与 `use_local_argmax_reduction=True` 组合被全面禁止(影响所有验证方法);FLy 与 synthetic rejection sampling 不兼容;`extract_hidden_states` proposer 被显式拒绝。

## 关联

同区域投机解码:与 DFlash(对比对象)、DSpark/MTP(MRV2 proposer)、block verification(`rejection_sample_method` 的其他取值)并存;本批索引中 #51655 为 DFlash 接入 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
