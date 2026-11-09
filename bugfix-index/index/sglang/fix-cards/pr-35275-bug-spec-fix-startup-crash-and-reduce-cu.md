# PR #35275: [Bug][Spec] fix startup crash and reduce CUDA graph memory usage for speculative adaptive

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#35275](https://github.com/sgl-project/sglang/pull/35275) |
| 作者 | Deleter-D |
| 合并日期 | 2026-08-27 |
| labels | run-ci, speculative-decoding |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
adaptive 投机解码三处问题:logits 缓冲过小启动崩溃、hook 错 backend 首次重放 IMA、capture 每趟新开流致几十 GB 冗余显存,分别修复。

## 问题现象
开启 `--speculative-adaptive` 后出现三个独立问题:
1. 启动崩溃:server 无法启动(logits 共享缓冲过小)。
2. 首次 CUDA graph replay 时 IMA(illegal memory access)。
3. CUDA graph 显存冗余达几十 GB(所有段读回 `inactive`),默认参数下 14.4 GB,作者自用配置下 44.3 GB。

## 根因
正文逐条披露:
1. `ModelRunner.max_decode_logits_rows()` 按静态 `decode_num_tokens_per_req()` 给共享 logits 缓冲定尺寸,而 adaptive 投机解码会重放为更大 draft-token 宽度构建的 runner,缓冲过小导致启动失败(问题在 #30549 提及)。
2. `DecodeCudaGraphRunner.capture_one_shape` 从 `self.model_runner.attn_backend` 而非本次 capture 实际使用的 `capture_prepare` 返回的 `attn_backend` 上查找 `on_after_cuda_graph_warmup`。ModelRunner 自有 runner 两者是同一对象,但 adaptive 会构建绑定各自新建 backend 的额外 runner,`model_runner.attn_backend` 仍指向活跃那个;hook 错 backend 使本该在 warmup 后升级的 metadata(raw→full、FlashMLA scheduler meta)被冻结进 graph,首次 replay 即 IMA。
3. 每次 capture pass 打开 `graph_capture()` 时未带 stream,于是每趟都在新流上 capture;caching allocator 按 stream 划分 graph pool 的段,导致每趟重复预留 MoE/DeepEP scratch 而不复用之前 pass 遗留的 inactive 段——candidate step 数 × 三个 runner 个 pass,累计几十 GB。

## 修复方案
1. `model_runner.py`:`max_decode_logits_rows()` 改为取 `max_speculative_num_draft_tokens()` 与 adaptive 各 candidate step(+1)中的最大宽度定尺寸(与 #30554 相同修法)。
2. `decode_cuda_graph_runner.py:1226-1237`:`post_warmup_hook` 改从 capture 局部的 `attn_backend` 解析:

```diff
+                # Adaptive runners may own a different backend than model_runner.
                 post_warmup_hook = getattr(
-                    self.model_runner.attn_backend,
+                    attn_backend,
                     "on_after_cuda_graph_warmup",
                     None,
                 )
```

3. `runner_utils/pool.py` 新增 `get_or_create_global_graph_capture_stream()`(Resources 层命名流租约),`DecodeCudaGraphRunner.capture` 与 `PrefillCudaGraphRunner.capture` 的 `graph_capture(stream=...)` 统一使用;capture 严格串行,共享单流安全。另新增 CPU 单测 `test/registered/unit/model_executor/test_graph_capture_resources.py` 与 `test_model_runner_decode_rows.py`。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:所有走 `max_decode_logits_rows()` 的配置(logits 缓冲按更宽 draft 宽度取 max,非 adaptive 场景数值不变);所有 decode/prefill CUDA graph capture(统一到共享 capture 流,非 CUDA 平台 runner 保留自有流,`_single_rank_graph_capture` 测试 shim 同步适配);`capture_one_shape` 的 warmup hook 解析(仅 adaptive 额外 runner 行为变化)。精度:DeepSeek-V4-Flash EAGLE+adaptive 下 GPQA Diamond/AIME25/GSM8k before 88.13/98.33/96.89 → after 88.04/97.29/96.97。吞吐不变,显存 14.4→10.13 GB(默认参数)、44.3→16 GB(作者配置)。CI:Base 绿,Extra 与 AMD ROCm 7.2 红(原帖标注)。

## 关联
- [#30549](https://github.com/sgl-project/sglang/pull/30549)(问题提及处)
- [#30554](https://github.com/sgl-project/sglang/pull/30554)(问题 1 的相同修法)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
