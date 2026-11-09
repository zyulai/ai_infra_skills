# PR #43012: [Speculative] Support block verification for the DFlash family

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @maocheng23 |
| 合并 | 2026-10-08 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, speculative-decoding, run-ci, jit-kernel, bypass-fail-fast |
| Diff | pack diffstat 缺失;diff 覆盖启动门控、DSpark/DFlash accept 路径、CLI help、docs 与两个测试文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/43012 |

## 一句话结论

把 block verification 扩展到 DFLASH/DSPARK 采样草稿,acceptance length +3.1–3.3%,吞吐 +0.5–1.8%(AIME25)。

## 动机

#42297 为 linear EAGLE / EAGLE3 / NEXTN 添加了 `--speculative-use-block-verification`([Block Verification Accelerates Speculative Decoding](https://arxiv.org/abs/2403.10444) Algorithm 2),但 DFlash family(`DFLASH`、`DSPARK`)虽然已经通过同一 Triton chain sampler 验证采样草稿,启动时该 flag 却被拒绝。DFlash2 每步 propose 7 个 token,早期拒绝后有更多 prefix 可被 block verification 挽救,收益空间比 3-token MTP 更大。

## 优化点

四层接线,kernel 本身不变:(1) 启动门控(`speculative_hook.py`)接受 CUDA/ROCm 上的 DFLASH/DSPARK;NPU 仍拒绝(`sgl_kernel_npu` 的 chain sampler 无 block mode);仅 EAGLE/EAGLE3 仍自动开启 `--speculative-use-rejection-sampling`,DFlash 族采样路径总是自带 proposal distribution,不读该 flag。(2) 共享 accept 路径(`dspark_accept.py`)的 `AcceptSampling` / `accept_sampling` / `_accept_sampling_core` 增加显式 `block_verification` 关键字,置位时以 `chain_speculative_sampling_triton(..., block_verification=True)` 调 #42297 的 kernel。(3) 调用方:DSpark 经 `DSparkWorkerV2` → `TargetVerifyExecutor` → `accept_draft_tokens` 传入 all-sampling 与 mixed greedy/sampling 两分支;DFlash 经 `_selector_sampling_accept` 转发(服务 candidate-selector 与 LiLiCorr 采样草稿)。(4) greedy DFlash 草稿(plain DFlash、Domino、无采样的 LiLiCorr)的 proposal 是 point mass(`q(x_i)=1`、`r_i=prod p(x_j)`),token-wise 验证已达 per-prefix 上限,block verification 接受相同 prefix,故保留原 target-only verifier 并打 warning;ragged/cutoff verify 仍在 sampler 结果后截断(target-distributed 样本的 prefix 仍 target-distributed,正确性不受影响)。

## 关键改动

```text
python/sglang/srt/arg_groups/speculative_hook.py: block verification 白名单加入 DFLASH/DSPARK(CUDA/ROCm)
python/sglang/kernels/ops/speculative/dspark/dspark_accept.py: accept 路径透传 block_verification 到 chain sampler
python/sglang/srt/speculative/dspark_components/dspark_verify.py: accept_draft_tokens 全分支透传
python/sglang/srt/speculative/dflash_worker_v2.py: _selector_sampling_accept 透传;greedy 草稿打 no-effect warning
```

## 性能声明

单 NVIDIA GB300、TP1,target Qwen3.8-27B BF16,FlashInfer attention,CUDA graphs on,radix cache off,concurrency 8;AIME25 全部 30 题、thinking xhigh、temp 1.0 / top-p 0.95 / top-k 20、32,768-token 上限、3 个 base seed(协议同 #42297):

- DFlash2(`--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`,draft DFlash2):acceptance length 4.0091 → 4.1416(+3.31% pooled,三个 seed 均为正);correct drafts per verify step +4.40%;output throughput 1260.93 → 1283.17 tok/s(+1.76% pooled,per seed +2.75% / +2.67% / −0.29%,PR 注明不一致);boxed score 76/90 持平。
- DSpark(draft RadixArk/Qwen3.8-27B-DSpark,verify window 8):acceptance length 4.0134 → 4.1390(+3.13% pooled,per seed +3.02% / +2.42% / +3.96%);correct drafts per verify step +4.17%;output throughput 1296.61 → 1303.58 tok/s(+0.54% pooled,per seed +1.51% / +0.42% / −0.45%,运行期间同节点其他 GPU 有负载、噪声更大)。

sampler 无 kernel 改动,成本为 #42297 实测的 block-mode 成本;未 opt-in 的路径不变。

## 验证与回归风险

`test_block_verification_args.py`(DFLASH/DSPARK 接受且不联动 rejection sampling;NPU 拒绝;rejected 用例从 DFLASH 移至 NGRAM);GPU 测试 `test_dflash_family_accept_uses_block_verification`(r1=1/2、r2=1 构造,all-sampling 与 mixed-greedy 分支);本地 45 passed + 21 subtests,GB300 上 30 passed + 43 subtests(含 `test_dspark_kernel_parity.py`)。风险:吞吐增益 per-seed 有负值,增益主要落在 acceptance length;NPU 路径显式不支持;greedy 草稿下 flag 无效仅 warning。

## 关联

前序 #42297(linear EAGLE/EAGLE3/NEXTN 的 block verification 与 Triton chain sampler kernel)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
