# PR #52212: [ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Fangzhou-Ai |
| 合并 | 2026-08-16 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, rocm, ready |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 另有 2 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/52212 |

## 一句话结论

优化 in-tree Triton DeepSeek-V4 sparse-MLA decode kernel 以适配 gfx950/MI355X:专用 partial kernel + 最多 32 splits + graph-safe 自适应 work 选择,8k/1k serving A/B 输出吞吐 +2.75%、P99 TTFT -7.9%。

## 动机

DeepSeek-V4 的 sparse-MLA decode 在 gfx950 上此前复用 gfx942 的实现(1-16 split selector、通用 partial/ragged kernel),没有利用 MI355X 的硬件特性。目标是在不引入 AITER/Gluon 运行时依赖、不切换实现的前提下优化默认 in-tree Triton 路径(corrected AITER/Gluon 仅作性能 comparator)。

## 优化点

所有新运行时行为以 `_ON_GFX950` 显式门控;gfx942 保持原路径(只读 AOT 审计显示其 HSACO 与干净基线逐字节相同)。gfx950 侧改动:专用 sparse-decode partial kernel——把 448 个 NoPE 与 64 个 RoPE 维合并为一次 512 宽 QK dot,四个 128 宽 PV accumulator 避免寄存器溢出,guarded BK32 tile 处理 main/SWA 工作,满 64 项 compressed-cache 组拆成两次有序 BK32 更新,保留 masked tails、attention sink 与 invalid-index 语义,在固定 captured maximum grid 下按行选择 live work;workload-aware split selection 面向 device occupancy,短 workload 封顶 4 splits,低 batch 长上下文最多 32 splits(`_decode_gfx950_num_splits`);adaptive captured-graph split selection(reduce 忽略 stale scratch)+ 收窄 ragged views(graph buffer 视图按 source 宽度截断但保留 graph-stable base pointer,indptr 仍承载真实 NNZ);直接 BF16 输出;compressed-cache NaN-free fast path——由显式、默认 false 的 provenance 门 `_trust_dsv4_extra_cache_nan_free`(gfx950 + fp8_ds_mla + 无 KV transfer + 有 extra cache)控制,两个写入端把 UE8M0 code 封顶 254 并规范化 RoPE NaN,main cache 仍 scrub、invalid index 贡献零概率、block 地址乘法用 int64。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: gfx950 partial kernel、_decode_gfx950_num_splits、adaptive reduce(diff 截断)
vllm/models/deepseek_v4/amd/rocm.py: NaN-free provenance 门、graph capture metadata 标记、adaptive_splits 传递、ragged view 收窄
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py: SANITIZE_CACHE_NANS 写入端(UE8M0 封顶 254、RoPE NaN→0)
tests/kernels/attention/test_rocm_triton_attn_dsv4.py, tests/kernels/test_compressor_kv_cache.py: gfx950 边界/图重放/NaN 契约测试
```

## 性能声明

Kernel acceptance(单 256-CU MI355X,精确 40 形状矩阵 T∈{1,2,4,8,16,32,48,64} × context∈{4K,32K,100K,500K,1M},eager 与 warmed-graph 正确性 + ≤1.03 性能门):vs clean in-tree Triton 40/40 接受、worst ratio 1.000000;vs corrected AITER/Gluon 40/40、worst ratio 1.023918;1,920 个正确性比较全部通过(atol=rtol=0.02)。Canonical 8k/1k serving A/B(8x MI355X,TP8/DP1,并发 64,prefix caching 关闭,128 warmups + 640 scored requests,输入/输出长度数组逐字节相同):输出吞吐 1,155.242→1,187.054 tok/s(+2.754%);duration 510.652→496.967 s(-2.680%);median TTFT 451.565→445.356 ms(-1.375%);P99 TTFT -7.919%;P99.9 TTFT -4.822%;P90 TTFT 为唯一回归项(+4.092%)。GSM8K(前一代 pure-Triton source):1,319 例全部完成,strict accuracy 0.960576、flexible 0.959818。

## 验证与回归风险

57 个 attention 测试、4 个 compressor/cp focused 测试通过,pre-commit 与 Python 3.12 mypy 通过。新增测试覆盖:不可信 cache 默认 scrub(NaN 注入后输出零)、NaN-free provenance 门、ragged buffer view 按 on_gfx950 收窄、capture metadata 标记、gfx950 split 选择(32/4/1)、adaptive reduce 忽略 stale scratch、outer64 边界(extra_len 0-65、无效/越界索引)、CUDA graph 重放下长短 workload 切换。风险:NaN-free 路径依赖两个写入端的缓存契约(UE8M0 ≤254、RoPE NaN 规范化),KV transfer 存在时禁用;graph 内 adaptive splits 依赖固定 grid 与 stale scratch 语义;gfx942 经审计不受影响。

## 关联

Issue #41820(检查过无重复);#50566(可复用 split-K scratch 与输出拷贝消除)、#51714(可选外部 AITER/Gluon 路径)——与本 PR 目标不同,本 PR 优化默认 in-tree Triton 实现。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
