# PR #46186: [Perf] spec: radix top-k for speculative verify target probabilities

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jianc99 |
| 合并 | 2026-11-11 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci, jit-kernel, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/46186 |

## 一句话结论

`build_speculative_verify_target_probs` 改在 raw logits 上做 FlashInfer radix top-k、只对选中的 k 个值除温度,省掉整张 [rows, vocab] 的温度缩放矩阵与全词表 `torch.topk`:GB300 vocab 248,320 下 4096 行 14.2→3.21 ms、fp32 临时峰值减半,端到端 decode tok/s +4.0–4.1%。

## 动机

`build_speculative_verify_target_probs`(`srt/speculative/dflash_utils.py`)是 DFLASH(selector 与 plain drafts)、DSPARK、UNO dense verification 与 sampling-mask capture 构建 sampled-verify target 分布的公共函数(EAGLE 系、standalone 与 n-gram verify 自建,不受影响)。原实现先把整张 [rows, vocab] logits 除以温度再 `torch.topk`;Qwen3.8 的 248,320 词表在 GB300 上:1024 行(128 请求 x 8-token blocks)每 verify step 3.8 ms,4096 行(512 请求)14.2 ms,每个 TP rank 都要付;还多一个 [rows, vocab] fp32 临时量。本 PR 是 DFlash adaptive-verification 系列的 component 3。

## 优化点

- 新增 `sglang.kernels.ops.sampling.topk.radix_topk`,按平台一次性 resolve(与 renorm ops 同法):FlashInfer 可用时用其 radix top-k(deterministic 模式、sorted 输出);否则或 CPU 张量用 `torch.topk`;接受直至词表大小的任意 k。
- `build_speculative_verify_target_probs` 在 raw logits 上选 top-k,只把选中的 k 个值除以温度——正温度保持 logits 顺序,概率不变。
- `srt/models/dflash.py` 里 DFlash2 selector 原有一份私有的 FlashInfer 调用(`_radix_topk`),改为使用共享 op;`radix_topk` 只读 dense rows(非连续输入会被拷贝),padded vocab 尾部仍以 mask 排除。
- 不变部分:per-row top-k mask、softmax、top-p renormalization、dense scatter;无 top-k 路径计算相同值。
- tie 行为:同一行 top-k 边界上并列时,保留哪些是确定性的(给定 batch),但可能与 `torch.topk` 的选择不同,且会随行数变化(行数决定 FlashInfer kernel 路径);保留 token 数与概率值相同。各 rank 看到相同 logits 时选出相同 ids(与 `torch.topk` 一致)。

## 关键改动

```text
python/sglang/kernels/ops/sampling/topk.py(新增): radix_topk(FlashInfer radix select,回退 torch.topk),注册为 sampling.radix_topk kernel
python/sglang/srt/speculative/dflash_utils.py: build_speculative_verify_target_probs 在 raw logits 上 radix_topk、只缩放选中值;scaled_logits 整表缩放移入需要的分支
python/sglang/srt/models/dflash.py: DFlash2 selector 删除私有 _flashinfer_top_k/_radix_topk,改用共享 radix_topk
test/registered/kernels/ops/sampling/test_radix_topk.py、test/registered/unit/spec/test_verify_target_probs.py(新增)
```

```python
        # Sparse exact path for joint top-k/top-p filtering, then scatter to dense.
        if 0 < max_top_k < vocab_size:
            # A positive temperature keeps the order of the logits, so select on
            # the raw logits and scale only the selected values.
            topk_logits, topk_indices = radix_topk(next_token_logits, max_top_k)
            topk_logits = topk_logits / expanded_temperature
```

## 性能声明

- 微基准(GB300,vocab 248,320,top_k 20,top_p 0.95,temperature 1):rows 512: 2.07→0.67 ms(峰值临时内存 0.95→0.47 GiB);1024: 3.77→1.04 ms(1.89→0.95 GiB);2048: 7.17→1.76 ms(3.79→1.90 GiB);4096: 14.2→3.21 ms(7.58→3.79 GiB)。128 行以下 build 受 launch 限制,前后都约 0.5 ms。
- 端到端(Qwen3.8-27B + DFlash2 draft(block 8)、static DFLASH、TP1、GB300、`--max-running-requests 128`、SPEED-Bench first turns、T=1 / top_p 0.95 / top_k 20、最多 4096 输出 token;before/after 在两块 GPU 上交错):decode tok/s c64: 5863→6099(+4.0%),c128: 6908→7189(+4.1%)。acceptance length 不变(c128 两侧 3.58–3.61)。512 请求(TP4)时每 rank 每 verify step 约省 11 ms(据微基准)。c128 时 running cap 等于并发,约一个请求排队,#46182 显示该设置在此档位比更大 cap 读数高 7–15%(两侧同样),c128 只在表内比较。
- 队列行为:约 1 s GPU work 排队时该调用约 1 ms 返回 host,不 synchronize,overlap scheduling 不受影响。

## 验证与回归风险

- 新 `test/registered/kernels/ops/sampling/test_verify_target_probs.py`:与前实现(top-k 前整表除温度)bitwise 相等——top-k 20/2048/2049/4096、uniform 与 mixed batch 含 greedy 行、有无 top-p;masked 到有限 logits 少于 k 的行;mixed batch 中每行自身边界上的 tie(精确 kept 数、合法 top-k、相同概率值、同 batch 重跑结果相同);温度除法制造的 tie;`torch.topk` fallback;`radix_topk` 对 `torch.topk`(fp32/bf16、连续与 strided、k 至 4096)。
- 既有测试通过:DFlash logits 与 selector、DSpark kernel parity 与 accept、speculative sampling、UNO sparse sampling、block verification、无 FlashInfer 导入。GPU 测试与基准在原 source branch 上跑过,upstream 更新(6a319e7b)后未重跑。
- 已知代价:compact(ragged)DSPARK verification 仍为其 strided buffer 的全部 bs x W 行构建概率;零 logit 的 padding 行全为 tie,radix kernel 代价约是普通行 4x(1024 行一半 padding 2.0 ms,无 padding 1.0 ms,改动前 3.8 ms);限制到 verified rows 需 logit 调整在 packed rows 上运行,留作 follow-up。
- 语义冲突:#41916(同一函数 sparse top-k/top-p 路径改在全词表分布上做 top-p,依赖本 PR 移除的 `scaled_logits`)、#43374(同一函数 dense top-p 分支,相差一个 hunk)、#44866(padding 行改 point mass 填充,上述 padding 代价未在该填充下测量)。UNO 保留自己的 FlashInfer top-k 调用,可另行迁移。

## 关联

#46182(DFLASH adaptive-verification 集成与全系统结果)、源分支 jianc99/sglang#3(component 3 原实现与单独 review);语义冲突/相邻:#41916、#43374、#44866。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
