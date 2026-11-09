# PR #51994: [Bugfix][Model] Fix DiffusionGemma silently freezing attention mask under CUDA graph replay

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51994](https://github.com/vllm-project/vllm/pull/51994) |
| 作者 | fjosw |
| 合并日期 | 2026-09-30 |
| labels | bug, nvidia, ready |
| 热度 | 👍 2 · 💬 9 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
DiffusionGemma 的 per-request causal 标志经 bool→int32 每次重建,FULL CUDA graph 永读捕获时旧值;改 int32 原地更新修复。

## 问题现象
默认优化级别 O2 会把 cudagraph_mode 解析为 FULL_AND_PIECEWISE(其 decode_mode() 为 FULL),因此 `vllm serve google/diffusiongemma-26B-A4B-it` 不加任何参数,每个 denoise/decode 步都跑在 FULL cudagraph 下;任何经已捕获 FULL graph 服务的请求,其 causal/bidirectional 模式被冻结为捕获(warmup)时该 slot 的模式,贯穿 graph 整个生命周期,且是静默的。作者未能把观测到的输出损坏明确归因于本 bug:E2E FULL-vs-PIECEWISE 对比不确定(denoising 自身 run-to-run 方差同量级),故改用 capture/replay 测试直接复现机制。H100 (SM90) 上回归测试在 main 失败(replay 返回 stale `[0, 0, 0, 0]` 而非更新后的 `[1, 1, 1, 1]`),patch 后通过。

## 根因
正文披露:`DiffusionGemmaModelState._causal_buf` 是 bool 张量,而 `FlashAttentionMetadataBuilder.build()` 对非 int32 causal 张量执行 `causal.to(torch.int32)`,每次调用分配新张量;FULL-mode graph 绑定捕获时看到的地址,此后永远读那个被遗弃(orphaned)的张量,`_causal_buf` 的后续更新永远到不了 graph。

## 修复方案
- `vllm/model_executor/models/diffusion_gemma.py`:`_causal_buf` 构造即 `dtype=torch.int32`;更新改为 `.copy_()`、padding 填 `0`,走 build() 的 dtype 匹配快速路径,不再重分配。
- `vllm/v1/attention/backends/flash_attn.py`:对非 int32 的 per-request causal 张量由静默 cast 改为 `raise ValueError`(说明在此 cast 会每次分配新张量、破坏 FULL CUDA graph replay,应把缓冲区分配为 int32)。
- `triton_unified_attention.py`:`per_seq_causal_ptr` 注释由 bool 更新为 int32。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围(默认 O2 即触发,非 opt-in)。
- 回归面:DiffusionGemma 的 attention metadata 准备与 FlashAttention metadata build;其他向 `build()` 传非 int32 causal 张量的调用方从静默 cast 变为显式报错(行为收紧,防止再引入此机制)。

## 关联
- #45163(原始 FULL-cudagraph 路径;正文 tag 其作者 @LucasWilkinson)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
