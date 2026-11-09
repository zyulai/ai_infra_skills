# PR #28265: [AMD] refactor sparse MLA decode kernel for Deepseek V4 triton backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-danli103(协作 @yichiche、@Raiden-Makoto) |
| 合并 | 2026-06-15 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci |
| Diff | 素材未提供 diffstat(diff 以删除文件为主,2 个删除文件完整展示,另有 4 个文件因体积省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/28265 |

## 一句话结论

移除 Triton sparse MLA decode 在 bs [33,255] 的 2-phase 回退、统一走融合 kernel,高并发 8k/1k TP8 ITL 提升 63-64%。

## 动机

Triton sparse MLA decode kernel 之前的 dispatch 逻辑基于 kernel 级 benchmark,对 decode batch size 在 [33, 255] 的情形回退到 2-phase(先 gather、后 attention 的分离实现)。但真实 E2E 测试中,该 2-phase 路径在高并发(8k/1k,TP=8,concurrency 64/128)下造成显著性能退化,原因:额外的 `gathered_kv` 中间 buffer 分配、额外的 kernel launch 开销、gather 与 attention 无法融合。Profiling 显示 bs=64 时 `_unified_sparse_decode_kernel` 单独占 decode attention 时间约 65%,而 bs=4 时融合的 `_fused_gather_attn_dsv4_dual_scope_kernel` 只占约 29.6%——高并发下应强制走融合 kernel 路径。

## 优化点

本 PR 是 PR #26722(已合入 `amd/deepseek_v4` 分支)向主线的移植,内容:(1) 删除 2-phase 分离实现所在的 3 个文件(`triton_mla_kernels_decode_common.py`、`triton_mla_kernels_decode_dsv4.py`、`triton_mla_kernels_decode_splitk.py`),其中含被删除的 `_gather_dequant_dsv4_kernel`(FP8→BF16 gather+dequant)与 `_unified_sparse_decode_kernel`(对 gathered_kv 的 attention);(2) 简化 kernel dispatch——所有 dual-scope 场景都用融合 kernel:小 batch 用 fused split-K kernel(更好 GPU 利用率),大 batch / extend / prefill 用 fused no-splitk kernel,原先走 2-phase 的 [33,255] 区间改用 fused no-splitk;(3) 代码清理:内联 `_bucket_total_tokens`、抽出共享函数 `_decide_splitk_dual_scope`、删除死代码。

## 关键改动

```text
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py:
  删除(2-phase 的 attention/chunking/splitk 入口)
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py:
  删除(2-phase 的 DSV4 gather+dequant 与 dispatch,含阈值常量)
python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_splitk.py:
  删除(独立的 split-K attention 实现)
(其余 4 个文件——融合 kernel 与 dispatch 侧改动——素材未展示)
```

被删文件中的 autotune 配置注释(CDNA4/gfx950,MI355X)节选:

```python
# CDNA4 (gfx950) Optimized: Added high-performance configs for MI355X
# Best config for h_q=128, large topk: BLOCK_H=64, BLOCK_N=256, num_warps=8
```

## 性能声明

8k/1k、TP=8、concurrency 64/128(即 decode batch 落在原 2-phase 区间的高并发用例):ITL 提升 63-64%,TTT 提升 32-38%;其余用例性能与之前保持一致。PR 未注明具体硬件型号(标题为 AMD,被删文件注释指向 CDNA4/gfx950 的 MI355X 配置)与 benchmark 工具;正文仅附结果截图,无逐项数字表。

## 验证与回归风险

精度:GSM8K 约 0.939 / 0.942(两轮确认),PR 未注明所用模型、硬件与采样参数(仅附截图)。单元/回归测试一节为空,checklist 各项未勾选。回归风险:删除的 2-phase 路径不再可用,bs [33,255] decode 全部走 fused no-splitk kernel,行为变化集中在此区间;`triton_mla_kernels_decode_*` 模块的删除会影响任何仍 import 这些模块的调用点(diff 中可见 dsv4 模块 import 了 common 的 chunked/splitk 入口);被删文件含 `TRITON_CACHE_DIR` 本地缓存目录设置逻辑,随删除一并消失。CI:PR Test (Base) 排队中、PR Test (Extra) 失败。

## 关联

本 PR 内容与 PR #26722 相同(后者已合入 `amd/deepseek_v4` 分支)。同区域(sparse MLA decode / DSV4 triton backend)其他 PR 编号素材未列出。
