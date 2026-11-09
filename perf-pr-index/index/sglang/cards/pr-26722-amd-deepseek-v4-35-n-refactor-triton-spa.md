# PR #26722: amd/deepseek_v4 35/N refactor Triton sparse MLA decode dispatch kernel to always run in a fused kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-danli103(Co-author: @yichiche, @Raiden-Makoto) |
| 合并 | 2026-05-29 |
| 分类 | Attention / Kernel(tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 2 个文件,另有 4 个文件因体积未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/26722 |

## 一句话结论

DeepSeek V4 sparse MLA decode 全部改走 fused kernel,高并发下 ITL 提升 63-64%、TTT 提升 32-38%。

## 动机

Triton sparse MLA decode kernel 的 dispatch 逻辑此前依据 kernel 级 benchmark,在 decode batch size ∈ [33, 255] 时回退到 2-phase(先 gather 后 attention 的分离实现)。但真实 E2E 测试(8k/1k、TP=8、cc=64/128)显示该路径在高并发下显著劣化,原因包括:额外分配 gathered_kv 中间缓冲、额外的 kernel launch 开销、gather 与 attention 无法融合。Profiling 显示 bs=64 时 `_unified_sparse_decode_kernel` 单独占 decode attention 时间约 65%,而 bs=4 时 fused 的 `_fused_gather_attn_dsv4_dual_scope_kernel` 仅约 29.6%,因此决定在高并发下也强制走 fused kernel。

## 优化点

删除 2-phase 实现的 3 个文件(`triton_mla_kernels_decode_common.py`、`triton_mla_kernels_decode_dsv4.py`、`triton_mla_kernels_decode_splitk.py`),这些路径不再被调用。简化 kernel dispatch:所有 dual-scope 场景统一使用 fused kernel——小 batch 用 fused split-K(更好的 GPU 利用率),大 batch / extend / prefill 用 fused no-splitk,原 [33, 255] 回退区间改用 fused no-splitk。附带代码清理:内联 `_bucket_total_tokens`、抽取 `_decide_splitk_dual_scope` 为共享函数、移除死代码。该 PR 属于 amd/deepseek_v4 集成分支与 main 的并行合入(merging to main is WIP)。

## 关键改动

```text
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py: 删除(2-phase 公共 kernel 与工具)
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py: 删除(DSV4 gather+dequant kernel 与 decode 入口)
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_splitk.py: 删除(2-phase split-K 路径)
```

## 性能声明

8k/1k、TP8、cc=64/128(batch size 落在原 [33,255] 回退区间)的场景:ITL 提升 63-64%,TTT 提升 32-38%;其余场景性能保持与之前一致。精度:gsm8k 约 0.939 / 0.942(两轮确认)。GPU 型号与模型 checkpoint 名称 PR 未注明(上下文为 amd/deepseek_v4 集成分支)。

## 验证与回归风险

精度依据为 gsm8k 两轮测试;pack 显示 CI 因缺少 `run-ci` label 未运行。风险:batch size [33,255] 的 decode 从"kernel 级 benchmark 认为更优"的 2-phase 改为 fused,依赖其 E2E profiling 结论;三个文件被直接删除,不再保留 2-phase 回退路径;删除的文件还包含 chunked attention、splitk wrapper 等公共工具,其他调用方若存在会直接报错。

## 关联

前序 Day-0 PR #23600、#23608(amd/deepseek_v4 集成);同分类(Attention / Kernel)另有 PR #11722(mrope Triton kernel),但区域不同。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
