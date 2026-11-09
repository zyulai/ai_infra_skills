# PR #33722: [KDA] Fused-accept state advance for FlashInfer KDA MTP verify

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-08-31 |
| 分类 | Speculative Decoding |
| Label | run-ci, run-ci-extra, linear-attention, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/33722 |

## 一句话结论

利用 FlashInfer `recurrent_kda` 的 `num_accepted_tokens` 参数把 KDA MTP verify 的 accepted-state 提交内联进内核,砍掉每轮全层 SSM scatter,B=16 verify 阶段省 62.5% 时间。

## 动机

FlashInfer 路径(SM100)的 KDA MTP verify 此前是两阶段:verify 内核把每个 draft token 的 post-state 写入 `intermediate_ssm` scratch,之后 `commit_mamba_states_after_verify` 把 accepted checkpoint scatter 回 persistent pool(`temporal`)——对每个请求、每轮都是一次全 recurrent-state 写,规模为 `num_kda_layers x HV x V x K x 4B`(HV=16、K=V=128 fp32 时约 4 MB/层/请求),外加一次内核 launch 与 host dispatch,全部落在 bs=1 MTP 的 inter-phase seam 上。FlashInfer 的 `recurrent_kda` 本就支持 `num_accepted_tokens`("fused speculative decode")让内核直接从 `ssm_state_indices[n, nat-1]`(上一轮最后一个 accepted token 的 checkpoint)自种子,但 sglang 从未传入。

## 优化点

一个 commit,由 `SGLANG_OPT_KDA_FUSED_ACCEPT_STATE` 环境变量开启(默认关闭,不影响任何默认路径)。四项改动:其一,verify 的 `[N, T]` 状态索引从 batch 位置改为按请求的 mamba slot(请求生命周期内稳定)构建,上一轮 checkpoint 跨 batch 成员变化仍可寻址;CUDA graph pad 行 slot 为 -1,所有派生索引保持负值,`recurrent_kda` 原生视为 inactive,无需额外 padding。其二,种子拷贝移出 per-verify 路径:committed-pool -> scratch slot-0 的每轮拷贝取消,fresh 请求在 extend 时一次性 staging(accept 长度 1),续请求由内核从 `slot[nat-1]` 自种子,`nat` 从持久 per-slot `accept_lens_pool` gather——全部 device 侧操作,CUDA graph capture 安全。其三,commit 收缩为 conv + bookkeeping:`commit_mamba_states_after_verify` 只保留 conv-window rollback(FlashInfer 不管 conv)加一次记录本轮 accept 长度的小 scatter,全层 SSM scatter 消失。其四,fail-loud 启动门控:要求 FlashInfer KDA decode backend、`--disable-radix-cache`(committed SSM pool 在 verify 之间过期)、与 `--enable-linear-replayssm-spec` 互斥。索引构建 hoist 到共享 forward metadata:20 个 KDA 层只构建一次而非每层各建(每层省约 9.2us)。位精确性 by construction:内核种子值与 scatter 提交的值字节相同。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/kernels/kda_flashinfer.py: build_fused_accept_indices(slot 索引 + nat gather,padding 负值不变量);target_verify 接 fused_accept 参数
python/sglang/srt/layers/attention/linear/kda_backend.py: _can_fuse_accept_state 能力判定 + accept_lens_pool + extend 时 staging + 索引 hoist
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: update_mamba_state_after_mtp_verify fused 分支(仅 conv scatter + accept len 记录)
python/sglang/srt/layers/attention/mamba/mamba2_metadata.py: ForwardMetadata 增加共享 fused-accept 索引字段
test/registered/attention/unittests/hybrid_linear/test_kda_fused_accept_indices.py: 6 个 padding 不变量与 per-forward cache 单测
```

## 性能声明

A/B 在 SM100(B200,FlashInfer 0.6.14),Kimi-Linear-48B 形状(20 KDA 层,HV=32,K=V=128,draft window T=4,bf16 state);20 个 per-layer verify 调用捕获进 CUDA graph 以 replay 计时,post-verify commit eager 计时;5x100 次取 min:

- B=1:legacy 393.4 us -> fused 218.5 us,省 **175.0 us(44.5%)**
- B=4:720.8 -> 312.3 us,省 **408.5 us(56.7%)**
- B=16:1967.8 -> 737.8 us,省 **1229.9 us(62.5%)**

B=1 分解:20 个 verify 内核约 212 us;被移除的 per-layer committed-pool 种子拷贝约 8 us/层;per-layer 索引构建约 9.2 us/层;被替换的 SSM commit scatter 实测 18.3 / 30.9 / 110.8 us(B=1/4/16),对比约 5 us 的 accept-length 写入。

## 验证与回归风险

6 个单测 pin padding 负值不变量(padded slot 必须对所有 step 产生负索引,算术重排会静默激活 pad 行)与 `accept_lens_pool` gather 语义、per-forward cache 生命周期(cache 跨 forward 存活会用上一 batch 的 slot 种子下一 batch,是无任何报错的静默错误状态)。GPU parity(SM100,B200,FlashInfer 0.6.14,真实 `recurrent_kda`):legacy-vs-fused 两轮对比在输出、最终 accepted 状态、committed pool 未被 fused 修改上均为 `torch.equal`(非 allclose),B=1/8/32、T=4/6 通过;hoist 单测 20 层共享 vs 各自构建 bit-identical。风险面:默认关闭(env 开启);`temporal` 在 verify 之间变 stale,任何遗漏的读取方(radix tracking、ReplaySSM、PD hand-off)都会被门控排除,新增能力需重新评估;仅 chain 布局(accept_lens = last_correct_step_indices + 1);Kimi-K3 DSpark 路径仍走 scatter,fused-accept 扩展为计划中的 follow-up。

## 关联

前序:PR #30113(FlashInfer KDA decode + MTP backend,本 PR 优化其 verify 提交路径)、PR #26888(KDA target_verify 基础)。Kimi-K3 DSpark verify kernel 的 fused-accept 扩展为规划中的后续工作。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
