# PR #26312: [mtp] add rejection sampling for speculative decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @liyucheng09 |
| 合并 | 2026-06-20 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/26312 |

## 一句话结论

EAGLE/MTP 引入经典拒绝采样:按 draft 分布采样起草、以 min(1, p/q) 验收,接受长度 +10%,吞吐相对 MTP 再 +9.4-15.4%。

## 动机

此前 EAGLE speculative decoding 的整条流水线——draft 侧预测与 target 侧验证——都在用 greedy(argmax)采样。这意味着非 temperature=0 的请求要么无法用 MTP 加速、要么接受率被 greedy 起草压低。经典拒绝采样(classic rejection sampling)让 draft 用真实概率分布采样,并把验收准则从"target p 贪婪接受"改为 min(1, p/q),使 speculative decoding 在任意温度下保持目标分布不变的同时大幅提升接受长度。实验显示各任务接受长度显著改善。

## 优化点

三处机制改动。其一,draft 采样从 argmax 改为按分布采样:`renorm_draft_probs` 在 RS 模式下按请求温度缩放 draft logits 后 softmax(温度对齐 target 采样温度以提高接受率;对任意 q 均保持正确性),`fast_sample` 用 `torch.multinomial` 采出单 token;每步 draft probs 堆叠成 (bs, num_steps, vocab) 张量,经 draft CUDA graph runner(新增 draft_probs buffer 与 temperatures 拷贝)、overlap utils 的 `draft_probs_buf`、forward batch padding 全链路传递。其二,新增 Triton kernel `speculative_sampling_classic_kernel`(chain 模式,topk=1 链式验证):逐 slot 比较目标概率 p 与 draft 概率 q,接受条件为 `coin * q < p`(即 min(1, p/q) 准则),拒绝后从 residual `max(p - q, 0)` 分布两遍(cumsum + CDF 查找)采样补 token;全部 draft 被接受时直接从纯 p 采样。其三,启动侧强约束:仅 EAGLE/EAGLE3/NEXTN、`--speculative-eagle-topk=1`、draft/target 词表一致,与 accept threshold、确定性推理、multi-layer EAGLE 互斥;非 RS 路径零额外开销(temperatures 仅 RS 时拷贝)。

## 关键改动

```text
python/sglang/srt/speculative/reject_sampling.py(新):
  speculative_sampling_classic_kernel Triton 实现(链式验证 + residual 采样)
python/sglang/srt/speculative/eagle_worker_v2.py:
  draft_forward/_draft_extend_for_{prefill,decode} 按 RS 分派 fast_sample
python/sglang/srt/speculative/eagle_utils.py:
  eagle_sample 按 RS 选择 chain_speculative_sampling_triton
python/sglang/srt/speculative/spec_utils.py:
  renorm_draft_probs(温度缩放)/ fast_sample
python/sglang/srt/arg_groups/speculative_hook.py:
  启动校验(algorithm/topk=1/词表一致/互斥项)
```

```python
# 验收准则从 p 改为 min(1, p/q)
if coin * q < p:
    num_accept += 1
    ...
else:
    continue_verifying = 0
# 拒绝后从 max(p-q, 0) 采样
val = tl.where(diff > 0.0, diff, 0.0)
```

## 性能声明

测试条件:model=Qwen3.5-397B-A17B-FP8,tp=8, dp=4, bsz=64, outputlen=16k(无 RS 基线 / MTP / MTP+RS 三臂)。atcoder python 数据集:temperature=1 时吞吐 1,640.34 → 3,280.80 → 3,785.60 tok/s(MTP+RS 相对无 MTP +130.8%),接受长度 3.06 → 3.37(+10.0%),verify count -10.1%;temperature=0.6 时 1,650.60 → 3,496.49 → 3,662.92(+121.9%),接受长度 3.33 → 3.46。deepmath:temperature=1 时 2,365.02 → 4,587.43 → 4,929.50(+108.4%),接受长度 3.20 → 3.46;temperature=0.6 时 2,334.44 → 4,781.54 → 4,916.06(+110.6%)。GLM-5 + deepmath + temperature=1:1,226.78 → 2,538.90 → 2,642.54(+115.4%),接受长度 3.06 → 3.25。精度:Qwen3.5-9B 单测 gsm8k score 0.905、output_throughput 1,404.6 tok/s。

## 验证与回归风险

新增 e2e 测试 `test_eagle_reject_sampling.py`(Qwen3.5-9B,NEXTN 3 steps/topk 1/4 draft tokens,TP2,register_cuda_ci 2-gpu-large)。风险面:RS kernel 从全局 RNG 抽 coin,与 `--enable-deterministic-inference` 不兼容(启动即拒);draft_probs 全词表 fp32 buffer 使 draft graph buffer 与 D2H 流量按 vocab 大小增长(32B 级模型内存开销可观);hot-token/reduced draft vocab(d2t map)尚不支持,词表不一致直接 raise;final sampling 的退化分支(norm_sum==0)回退 VOCAB_SIZE-1;温度缩放 draft 分布改变了非 RS 用户的 draft 行为吗——不会,gated by flag,但 topk>1 路径与 tree verify kernel 的组合被显式禁用。

## 关联

同区域(Speculative Decoding):本批次 #32219(spec-v2 host-seam)、#23958(非对称 PD-MTP)、#30393(HiCache draft 缓存)均属 spec 基础设施;diff 中引用 #26358(ROCm argmax tie-break 问题)。其余同分类编号 pack 未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
