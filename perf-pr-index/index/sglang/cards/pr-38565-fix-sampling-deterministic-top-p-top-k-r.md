# PR #38565: [Fix] sampling: deterministic top-p / top-k renorm when tp_size > 1 (TP rank divergence in DFlash/DSpark)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @gilfordting |
| 合并 | 2026-10-31 |
| 分类 | Speculative Decoding |
| Label | sgl-kernel, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/38565 |

## 一句话结论

TP>1 时改用 flashinfer `is_deterministic=True` 的 top-p/top-k renorm,消除跨 rank 概率分歧引发的 NCCL hang,TPOT 代价 ≤0.2%。

## 动机

flashinfer 默认 top-p/top-k renorm kernel 使用 float `atomicAdd` 汇聚部分和(top-p 长行走 AIR radix kernel,top-k 走 `RadixTopKRenormProbMultiCTA`),相同 logits 在不同 TP rank 上可产生最后几位不同的 `target_probs`。spec-verify 用共享采样 coin 对该输出做接受判断,coin 落在跨 rank 差距内时各 rank 提交不同 token,KV/radix 状态漂移,后续前缀匹配触发 NCCL collective hang。此前 #33549、#33289、#33614 用广播 rank 0 决策绕过,本 PR 直击根因并省去额外 collective。

## 优化点

新增 `sampling_renorm.py` 模块承载 renorm 策略:`renorm_deterministic()` 按 `tp_size > 1` 或 `--enable-deterministic-inference` 决定(注意用 tp_size 而非 attn_tp_size,DP attention/CP 下 attn_tp_size 为 1 但 rank 仍需一致),`SGLANG_RENORM_DETERMINISTIC` 环境变量可强制开/关;单 rank 保留 fast kernel。deterministic 变体调用 flashinfer `top_p_renorm_probs`/`top_k_renorm_probs` 的 `is_deterministic=True`(整数直方图 AIR top-p、定点求和 radix top-k,flashinfer#4831 引入,延迟不变);import 时用 `inspect.signature` 检测旧版 flashinfer,自动退化为 fast kernel 而非在首个请求上失败。sampler、`dflash_utils.py`、`eagle_utils.py` 三个调用点统一切换;MUSA/XPU 的 sgl_kernel 单 CTA kernel 本身确定,不加改动。

## 关键改动

```text
python/sglang/srt/layers/sampling_renorm.py: 新增策略与 deterministic renorm 包装
python/sglang/srt/layers/sampler.py / speculative/dflash_utils.py / eagle_utils.py: 调用点切换
python/sglang/srt/environ.py: SGLANG_RENORM_DETERMINISTIC 开关
test/registered/unit/sampling/test_renorm_policy.py: 策略单测(CPU)
```

```python
def renorm_deterministic() -> bool:
    if get_exec().deterministic.enable_deterministic_inference:
        return True
    return get_parallel().tp_size > 1
```

## 性能声明

kernel(H100,flashinfer 0.7.1,vocab 128256,99 次重复):top-p 32 行 flat 0.136 ms(nondet 99/99)→ 0.169 ms(0/99);256 行 flat 0.603→0.815 ms(top-p 增加 20~45%);top-k 两臂几乎相同(0.042/0.289 ms)。e2e(Llama-3.1-8B,H100,bench_serving 512-in/256-out,400 prompts,cc32):plain TP2 min_p=0.05 TPOT -0.0%~-0.1%,DFlash TP2 top_p=0.9 +0.2%(accept len 4.563/4.564),EAGLE3 TP2 +0.2%。复现实验:广播关闭 + fast kernel 时 4x H100、2x B200 全部 hang(190~610 s),rank-local top-p mismatch 0.4~0.9%;deterministic kernel 5 次全干净跑满 30 分钟(22.6k~35.4k 请求,276,000 次 top-p 调用 0 mismatch)。

## 验证与回归风险

`test_renorm.py` 新增回归测试:flat 256x128256 输入上各 deterministic renorm 重复 30 次 assert bit-identical(默认 kernel 30/30 失败),并断言 TP=2 配置下默认路径命中同一 kernel;新增 CPU 单测 `test_renorm_policy.py` 覆盖单 rank fast、多 rank deterministic、DP attention 下跟随 tp_size、deterministic-inference 强制。风险面:所有 TP>1 的采样路径(含 min_p)统一走确定性 kernel,top-p 有 20~45% kernel 级开销但 e2e 不可测;旧版 flashinfer(<0.7.x 的 top-k / <0.6.7 的 top-p)自动回退 fast kernel,不做硬失败。pack 显示 CI Extra 失败、AMD ROCm 10 待跑。

## 关联

前序 workaround:#33549、#33289、#33614(broadcast 方案,与本修复互补);上游依赖 flashinfer#2752(AIR top-p)与 flashinfer#4831(fixed-point deterministic)。同分类其他 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
