# PR #51855: [K3] support recoverssm for K3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-08-17 |
| 分类 | Speculative Decoding \| tier: major |
| Label | performance, ready, mrv2, verified, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 9 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/51855 |

## 一句话结论

RecoverSSM 用单 checkpoint 重建替代逐投机位全状态，commit kernel 快 3.27x、cache 容量 +10.97%。

## 动机

PR 正文 Purpose：新增 RecoverSSM，即 NVIDIA Model Runner V2 上针对 speculative KDA decoding 的 Kimi-K3 专用状态恢复路径，通过 `--use-replayssm` 启用，面向 Kimi-K3 搭配 DSpark 的服务，并支持 `mamba_cache_mode=align` prefix caching。正文指出：没有 RecoverSSM 时，KDA speculative decoding 会为每个投机位置物化一份完整的 recurrent state。

## 优化点

RecoverSSM 只保留一个 checkpoint 与紧凑的逐 token 记录，在每次 verify 后于 accepted position 重建状态。KDA verify 只读 checkpoint 不修改，为每个投机 token 记录 FP32 的 correction vector `u`、以及激活 dtype 的 key 与 raw gate；用 FP32 保存 `u` 以保持重建精度（评估配置中 key/gate 仍为 BF16），从而把 `v`、`beta` 移出 commit 路径、避免重算状态相关校正。verify 确定 accepted length 后，一个 Triton kernel 按递推 `S_p = S_0 * exp(sum g) + sum_j u_j outer (k_j * exp(...))` 逆序直接重建所需 checkpoint，只加载一次原始状态、不递归写中间状态，循环上界为运行时 accepted length 而非完整的八 token DSpark 窗口。align 模式下同一次 launch 还重建下一个 prefix-cache 边界。每个 KDA page 的布局为 `(convolution state, KDA checkpoint, FP32 correction records, key/gate records)`，恢复记录是临时的、不进入 prefix-cache 状态拷贝。metadata builder 给投机行分配一个 KDA checkpoint slot 并创建模型专用的 post-sampling committer。

## 关键改动

```text
vllm/models/kimi_k3/nvidia/kda.py: Kimi-K3 KDA 层与 RecoverSSM 状态恢复/commit 路径
vllm/config/vllm.py: 由 --use-replayssm 派生的 KDA RecoverSSM flag
vllm/config/cache.py: cache layout 相关配置
vllm/model_executor/layers/mamba/mamba_utils.py: state-layout helpers 与 commit 生命周期
tests/models/kimi_k3/test_kda.py: RecoverSSM 单元测试（含 18 个相关用例）
```

## 性能声明

Nsight Systems（batch 32、128 forced output tokens、每 GPU 80 个 decode steps、同一两节点 TP8 配置，聚合含 8 个小 batch 起始步）：KDA work per decode step 中 Verify 3.115 → 3.540 ms、Commit 4.793 → 1.464 ms、Total 7.989 → 5.085 ms；正文结论 "The direct reconstruction makes the commit kernel 3.27x faster and reduces total RecoverSSM KDA time by 36.34%. The FP32 correction write increases verify time by 13.66%, but commit time falls by 69.45%"。内存容量（固定 46.48 GiB cache budget）：Effective cache capacity ON 2,856,170 vs OFF 2,573,777 tokens（+10.97%）；Max 262k-request concurrency 10.895x vs 9.818x（+10.97%）。端到端精度（Kimi-K3 target + Kimi-K3-DSpark draft，TP8 两节点 GB300，七个 greedy draft tokens，block rejection sampling，FP8 MLA cache）：GSM8K 1,319 samples flexible 96.3609%(ON) vs 96.1334%(OFF)，strict 96.0576% vs 95.9060%；MRCR cold/warm 0.74879 vs 0.66791；Total DSpark acceptance 53.3514% vs 53.4062%。

## 验证与回归风险

Unit tests `pytest tests/models/kimi_k3/test_kda.py -v` 69 passed（18 个 RecoverSSM 用例覆盖八 token DSpark 窗口、部分/全接受长度、align 边界重建、两种卷积状态布局等）；pre-commit all hooks passed、mypy-3.12 passed。Scope 限于 NVIDIA Kimi-K3 KDA、MRV2、Kimi-K3-DSpark 加七个 draft tokens、align prefix caching；regular KDA prefill 与非投机 decode 路径不变，不新增 Mamba2 支持。正文说明独立初始化的 greedy serving 非 bitwise deterministic，故用 paired score 比较而非逐字一致；正文未说明对其他模型或硬件的影响。

## 关联

正文 Duplicate-work check 列出 #49847（面向 Mamba2）、#49887（面向 Gated DeltaNet）、#50140（WIP FlashInfer ReplaySSM kernel），并称无其他 open PR 实现该 Kimi-K3 KDA RecoverSSM 集成。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
