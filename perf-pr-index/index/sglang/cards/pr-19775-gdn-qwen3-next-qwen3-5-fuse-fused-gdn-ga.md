# PR #19775: [GDN][Qwen3-Next][Qwen3.5] Fuse fused_gdn_gating and fused_recurrent_gated_delta_rule_update in verify_target

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-03-06 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(diff 正文可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/19775 |

## 一句话结论

把 GDN verify 路径的两个 kernel 融合为一个,8k 输入/1.5k 输出下 TTFT 降低 11%。

## 动机

GDN(Gated DeltaNet)有三个关键 kernel:extend、decode、verify_target。当时 verify 路径先用 `fused_gdn_gating` 算 gating,再调 `fused_recurrent_gated_delta_rule_update` 做递归更新,两次 launch 且寄存器占用高。作者的第一个方案是直接复用为 decode 设计的单步 kernel `fused_sigmoid_gating_delta_rule_update`,但它缺少 verify 所需的三项能力——不写回状态的控制、中间状态缓存、tree attention 支持——累积导致精度从 0.83 崩到 0.35。因此需要一个真正支持 target verify 的融合 kernel。

## 优化点

扩展 `fused_sigmoid_gating_delta_rule_update` kernel(Triton),在 kernel 内直接从 `A_log/a/dt_bias/b` 计算 sigmoid gating,省掉独立的 `fused_gdn_gating` launch,并新增三个 constexpr 能力:`DISABLE_STATE_UPDATE`(verify 不写回最终状态)、`CACHE_INTERMEDIATE_STATES`(每步把 `b_h` 存入 `intermediate_states_buffer`,供后续步读取)、`HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`(每步按 `retrieve_parent_token` 找到父 token,从缓存加载父步状态,支持 EAGLE tree attention)。原 decode 单步路径行为不变(新增参数默认关闭);`gdn_backend.target_verify` 的接口从传预算好的 `g/beta` 改为传原始 `a/b/A_log/dt_bias`,gating 计算下沉进 kernel;Triton dispatcher 的 `target_verify` 从两步实现切换为该融合 kernel,FlashInfer MTP dispatcher 也改用统一入参(去掉 `a_raw/b_raw` kwargs 旁路)。 refactor 后 KDA 未来支持 MTP 时也可复用同一 kernel(`is_kda` 分支内已有对应 mask/stride 处理)。

## 关键改动

```text
python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py: kernel 增加 verify 三能力与 gating 内联
python/sglang/srt/layers/attention/linear/gdn_backend.py: target_verify 改传原始 a/b,延后 gating
python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py: target_verify 切到融合 kernel
python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py: 统一 MTP 入参
python/sglang/jit_kernel/tests/test_fused_verify_triton_gdn.py: N×T 精度矩阵测试
```

## 性能声明

Qwen3-Next-80B-A3B-Instruct,TP4,speculative NEXTN(3 步/topk 1/4 draft tokens),`--disable-radix-cache`,bench_serving random 8000 输入 / 1500 输出,200 prompts,并发 ~118:Mean TTFT 16,241.02 → 14,403.37 ms(-11%),Mean TPOT 12.16 → 10.75 ms,Mean E2E 25,176.97 → 22,195.69 ms,输出吞吐 3,600.74 → 4,074.37 tok/s,accept length 3.74/3.75 基本一致。精度:GPQA(198 例 repeat 8)均值 0.542,作者称无下降;GSM8K(lm_eval,main vs PR)flexible-extract 0.8643±0.0094 → 0.8537±0.0097、strict-match 0.8476±0.0099 → 0.8127±0.0107,PR 正文称"gsm8k has no drop",数字上 strict-match 有所下降,以原文数字为准。

## 验证与回归风险

新增 `test_fused_verify_triton_gdn.py`:9 个 (N,T) 组合(1/8/16 batch × 1/4/8 step)对两步参考实现对比,max_diff 4.88e-04~9.77e-04,fail_rate(>0.1) 0.00%;另有单步 decode 状态一致性检查(fail_rate < 0.01% 断言)。风险面:融合 kernel 的 tree-attention 状态加载逻辑只在该路径触发,decode 路径靠默认 constexpr 关闭保持原行为;FlashInfer 与 Triton 两个 dispatcher 的入参签名同时改动,任何其他调用 `target_verify` 的后端需要同步;GSM8K strict-match 的轻微下降值得在后续版本复核。

## 关联

同区域:GDN 系列(extend/decode/verify 三 kernel 的其他优化);KDA 后续支持 MTP 时可复用本 kernel。pack 未列出其他具体 PR 编号。
