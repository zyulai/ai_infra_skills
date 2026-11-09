# PR #44268: [Fix] moe: Fix --enable-fused-moe-sum-all-reduce outside CUDA and with LoRA hooks

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ZhengGong-amd |
| 合并 | 2026-10-30 |
| 分类 | MoE(major;正确性修复,非提速) |
| Label | run-ci |
| Diff | 素材未提供 diffstat(diff 含 2 个文件:fused_moe.py 与 test_fused_moe.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/44268 |

## 一句话结论

修复 `--enable-fused-moe-sum-all-reduce` 在非 CUDA/MUSA 平台与 LoRA `after_down` hook 下的错误输出,ROCm GSM8K 从 0% 恢复到基线。

## 动机

`--enable-fused-moe-sum-all-reduce`(#19672)让 Triton down GEMM 把专家求和直接写入 `out_hidden_states`、不写 `intermediate_cache3`;但 `_fused_moe_kernel_sequence` 只在 `_is_cuda or _is_musa` 时遵守该约定,并且忽略 LoRA 的 `after_down` hook。后果:在 HIP、XPU 及 fallback 路径上,combine 会对未写入的 `intermediate_cache3` 再做一次归约(错误结果);所有平台上 LoRA 加到 `intermediate_cache3` 的 down delta 被丢弃。在 ROCm 上,每个 `topk > 2` 的 Triton-runner MoE 层都会出错:Qwen3.6-35B-A3B 与 Kimi-Linear-48B 用 `--moe-runner-backend triton`、GLM-5.3 / MiniMax-M2.7 / Qwen3-Next / Qwen3.8-Flash-Next FP8 用 `SGLANG_USE_AITER=0` 时 GSM8K 均为 0%;gemma-4-26B-A4B(I=704)默认就走 Triton,仅开 flag 即坏。

## 优化点

三处修复,均为正确性语义,无性能目标:(1) 把 fused 情形的处理提到平台 dispatch 之前——`moe_combine_topk` 中 `use_fused_moe_sum_all_reduce` 分支优先于 `_is_cuda or _is_musa` 分支,使 HIP/XPU/fallback 与 CUDA/MUSA 行为一致;并按平台未融合 combine 的方式应用 `routed_scaling_factor`(aiter 路径除外,其 `moe_sum` 同样跳过该缩放);CUDA/MUSA 代码只是移动、未改语义。(2) 融合启用条件增加 `(not (hooks and hooks.after_down is not None))`:存在 `after_down` hook 时禁用融合,保住 LoRA delta;空的 `LoRAHooks()`(启用 LoRA 但无激活 adapter)仍保持融合。(3) 新增 `TestFusedMoeSumAllReduce`:flag-on 输出须匹配 flag-off(相对误差 < 2e-2),并用 mock 断言融合 kernel 确实发射,覆盖 routed scale None/2.5 x 无 hook/空 hook/`after_down` hook。修复前 6 个 `after_down` 用例以约 20% 相对误差失败。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:
  融合条件加 LoRA after_down 守卫;combine 分支重排(fused 先于平台 dispatch)
test/registered/moe/test_fused_moe.py: 新增 TestFusedMoeSumAllReduce
```

combine 修复(节选):

```python
if no_combine:
    pass
elif use_fused_moe_sum_all_reduce:
    if routed_scaling_factor != 1.0 and not _use_aiter:
        assert out_slice is not None
        out_slice.mul_(routed_scaling_factor)
elif _is_cuda or _is_musa:
    ...
```

## 性能声明

不追求性能变化。回归验证:gemma-4-26B-A4B-it,cookbook 命令,flag off,TP1,`sglang.benchmark.serving` random 1937/389,307 prompts,concurrency 61,4 轮取第 2-4 轮 median:total throughput 13870.9 → 13935.4 tok/s(+0.5%),TTFT 与 TPOT 持平。另注:flag 开启时 gemma-4 在 gfx950 上的吞吐比 flag-off 低约 60%(`acq_rel` atomics 所致),因此 ROCm 上该 flag 应保持关闭——这是对既有 flag 的性能注记,非本 PR 改动。

## 验证与回归风险

MI355X、main `b960aa903a`:`TestFusedMoeSumAllReduce` 在 `SGLANG_USE_AITER=0` 与 `=1` 下均通过(main 在两种配置下 18/18 全失败),`test_fused_moe.py` 其余用例通过。`sgl-eval` GSM8K(temperature 0,MI355X;前两行 1319 题、其余 200 题,`SGLANG_USE_AITER=0`,auto 解析为 Triton):

| 模型 | main flag-off | main flag-on | 本 PR flag-on |
|---|---|---|---|
| Qwen3.6-35B-A3B | 94.09 / 94.16 | 0.00 | 94.31 |
| Kimi-Linear-48B-A3B(routed scale 2.45) | 91.28 / 91.96 | 0.00 | 92.42 |
| GLM-5.3 FP8,TP4 | 51.5 | 0.0 | 49.0 |
| MiniMax-M2.7,TP2 | 96.0 | 0.0 | 96.0 |
| Qwen3.8-Flash-Next-FP8 | 96.5 | 0.0 | 96.0 |
| Qwen3-Next-80B-A3B-FP8 | 96.0 | 0.0 | 96.0 |

GLM-5.3 的 51.5→49.0 差距在该 aiter-off 配置下与 flag 无关(模型在多数答案上循环),属 n=200 噪声。已知边界:Block-FP8 MoE + `SGLANG_USE_AITER=1` + triton runner 在 main 上 flag-off 即坏,由 #41636 单独修复。CI:Base 通过,Extra 与 AMD ROCm 10 失败。回归面:改动位于所有平台的 Triton MoE combine 公共路径,重排后 aiter 路径保持不加 `routed_scaling_factor` 的原语义。

## 关联

依赖/修复 flag 引入 PR #19672 的行为;Block-FP8 + AITER 问题由 #41636 单独处理;由 AMD Hyperloom 发现。同分类其他 PR 编号素材未列出。
