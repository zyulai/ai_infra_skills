# PR #38565: [Fix] sampling: deterministic top-p / top-k renorm when tp_size > 1 (TP rank divergence in DFlash/DSpark)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38565](https://github.com/sgl-project/sglang/pull/38565) |
| 作者 | gilfordting |
| 合并日期 | 2026-10-31 |
| labels | sgl-kernel, run-ci, jit-kernel |
| 热度 | 👍 1 · 💬 25 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
flashinfer top-p/top-k renorm 用 float atomic 导致 TP rank 间结果分歧,投机验证Commit 后 NCCL 集合通信挂死;TP>1 时改走 `is_deterministic=True` kernel 修复。

## 问题现象
TP=2 DFlash(Llama-3.1-8B + `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`,`temperature=1.0`、`top_p=0.9`、32 路多轮并发、fp8 KV)在关闭所有广播集合通信后,fast kernel 每次运行都复现挂死:H100 × 4、B200 × 2 共 6 次运行全部在 190~610 秒内 hang。记录到的一致故障链是:`target_probs` 在 logits、`prefix_lens`、RNG 完全相同的情况下跨 rank 不一致 → 某个 rank 的 `bonus`/`out_tokens` 不同 → 下一步 `draft_tokens` 分歧 → hang。rank 级 `top_p_renorm_prob` 重复调用不匹配率 0.4~0.9%;切到确定性 kernel 后 5 次运行(22.6k~35.4k 请求)零分歧、276,000 次 rank 级调用零不匹配。sampler 的 `min_p` 路径同样调用这些 kernel(`SGLANG_SYNC_TOKEN_IDS_ACROSS_TP` 默认关闭),也在受影响面内。早期合成复现未触发是因为用了 `top_p=1.0`,直接跳过该 kernel。

## 根因
PR 正文披露:`sgl_kernel.top_p_renorm_prob` 和 `top_k_renorm_prob` 转发给 flashinfer——top-p 在行长超过 2048 时用 AIR radix kernel(flashinfer ≥0.6.7,flashinfer-ai/flashinfer#2752),其默认直方图是 float atomic;top-k(`RadixTopKRenormProbMultiCTA`)用 float atomic 池化保留质量。因此对相同 logits,各 TP rank renorm 结果可在末位比特上不同。每个消费方(普通 sampler、spec-verify)对照一个共享确定性 coin 采样并把结果提交到 per-rank KV/radix 状态;coin 落在跨 rank 差异缝隙里时,部分 rank 接受/部分拒绝,KV 状态漂移,后续 prefix match 使 NCCL 集合通信挂死。flashinfer-ai/flashinfer#4831(0.7.x,sglang pin 0.7.2)已在时延不变的前提下加入定点 `is_deterministic=True` 变体,本 PR 据此修根因。

## 修复方案
新增 `python/sglang/srt/layers/sampling_renorm.py`:提供 `renorm_deterministic()` 策略与 `top_p_renorm_prob`/`top_k_renorm_prob`(sampler)及 `spec_top_p_renorm_prob`/`spec_top_k_renorm_prob`(verify),在需要时以 `is_deterministic=True` 调 flashinfer,否则回落原 fast kernel。策略表:单 rank 用 fast(无人需要对齐);`tp_size > 1`(非 `attn_tp_size`,因 DP attention/context parallel 下 `attn_tp_size` 可为 1)用 deterministic;`--enable-deterministic-inference` 强制开启;`SGLANG_RENORM_DETERMINISTIC=1/0` 显式覆盖。flag 存在性在 import 时用 `inspect.signature` 检查一次,旧版 flashinfer 降级到 fast kernel 而非在首个请求上崩溃。三个 renorm 调用点 `layers/sampler.py`、`speculative/dflash_utils.py`、`speculative/eagle_utils.py` 改为从新模块导入;`environ.py` 注册新环境变量;MUSA/XPU 无 flashinfer,`sgl_kernel` 单 CTA kernel 本身确定。修复放在 sglang 而非 sgl_kernel,因为后者以 pin 死的预编译 wheel 发布,CI 无法测试、要到下一个 kernel 版本才能交付。关键策略逻辑:

```python
def renorm_deterministic() -> bool:
    if get_exec().deterministic.enable_deterministic_inference:
        return True
    return get_parallel().tp_size > 1
```

回归测试:现有 kernel 正确性测试走新模块;新增测试对 256×128256 flat 输入连续调用 30 次断言 bit-identical(默认 kernel 30/30 失败);CPU 单测 `test_renorm_policy.py` 覆盖策略各分支。端到端 TPOT 差异 ≤0.2%(量级 0.01 ms),accept length 不变。

## 影响与回归风险
- 回归面:所有 top-p/top-k renorm 调用(sampler 与投机验证)在 `tp_size > 1` 或 `--enable-deterministic-inference` 下换用确定性 kernel——top-p kernel 本身增加 20~45%(0.1~0.8 ms 的 kernel 上),端到端实测无可测成本;单 rank 路径与 MUSA/XPU 路径不变。旧版 flashinfer(<0.7.x 的 top-k、<0.6.7 的 top-p)自动降级 fast kernel。`_sync_token_ids_across_tp` 及 `spec_tp_sync.py` 注释更新。与 #33614 互补:广播仍是针对未来非确定性算子的保险,本 PR 消除已发现的这一个,并使 logprobs、sampling mask 等 `target_probs` 其他消费方可复现。

## 关联
- #33549、#33289、#33614(此前的广播 workaround)
- flashinfer-ai/flashinfer#2752、flashinfer-ai/flashinfer#4831(上游 kernel 来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
