# PR #41400: feat(kda): integrate FlashInfer prefill checkpoints

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kpham-sgl |
| 合并 | 2026-10-09 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/41400 |

## 一句话结论

为 KDA prefill 接入 FlashInfer CuTe DSL 后端并支持 radix checkpoint,单序列 kernel 1.3-1.8x、chunked 形状 3.2x。

## 动机

FlashInfer 0.7.0(#5021)在 CuTe DSL KDA prefill 中新增了 indexed、floor-aligned 的 state checkpoint,这正是 SGLang radix prefix caching 对线性注意力状态所需要的能力。此前 SGLang 的 KDA prefill 只有 Triton/Helion/flashkda 等 kernel,FlashInfer 侧仅有 decode/verify 包装;没有 checkpoint 支持,KDA 前缀缓存路径缺少高效的 prefill 后端。

## 优化点

新增 opt-in 的 `--linear-attn-prefill-backend flashinfer` 后端(限 BF16、128-dim safe-gate KDA、SM100/SM103)。核心机制是:新增的 adapter 直接调用 `flashinfer.kda.recurrent_kda(..., backend="cute-dsl")`,由 FlashInfer 负责调度与 workspace 分配;SGLang 侧则在每个 tracked batch 执行前,用 `build_flashinfer_kda_checkpoint_plan` 把调度器生成的 radix-cache track 边界映射成 checkpoint 目的地索引(-1 表示跳过),使 prefill 期间产出的 checkpoint 可直接写入 FP32 state pool 供 radix 缓存复用。同时统一 beta 契约:KDA 模型在所有阶段传 raw beta logits,激活(sigmoid)下放给各 kernel adapter,移除模型里按 forward mode 预先 sigmoid 的 backend 相关分支。兼容性方面,setup 阶段拒绝不兼容的 checkpoint interval、TBO 与 full prefill CUDA graph;单 token 的 packed prefill 回退 Triton(FlashInfer packed API 不支持)。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/kernels/kda_flashinfer_prefill.py: 新增文件;FlashInfer CuTe DSL KDA prefill adapter + checkpoint 计划构建
python/sglang/srt/layers/attention/linear/kda_backend.py: dispatcher 接入 flashinfer prefill 后端、_validate_flashinfer_kda_prefill 校验、raw-beta 契约
python/sglang/srt/models/kimi_k3.py / kimi_linear.py: 移除模型侧 beta 预 sigmoid,改传 raw logits
test/registered/attention/test_kda_prefill_flashinfer.py: 新增文件;6 个 checkpoint/layout 用例 + 2 个数值用例
```

```python
# kda_flashinfer_prefill.py: 只写 radix-tracked 的 checkpoint,其余目的地置 -1 跳过
checkpoint_destinations = [-1] * checkpoint_starts[-1]
for row, tracked in enumerate(track_mask):
    if not tracked:
        continue
    relative_track_len = track_lens[row] - prefix_lens[row]
    if relative_track_len % chunk_size == 0:
        continue
    completed_chunks = relative_track_len // chunk_size
    checkpoint_destinations[checkpoint_starts[row] + completed_chunks - 1] = (
        num_tracked_checkpoints
    )
    num_tracked_checkpoints += 1
```

## 性能声明

Speed Tests 为 wrapper 集成时期的历史测量,PR 明确说明直接调用 API 后"性能与 D2H 行为需在最新 head 重测":

- Kernel-only KDA prefill(checkpoints off),K3 shape 单 TP8 rank(12 heads x 128),单张 GB300,bf16 输入/fp32 state,5 次中位数(ms):16k 序列 Triton 1.18 vs FlashInfer 0.87;128k 单序列 7.76 vs 4.36;128k 按 8x16k chunk 的 chunked prefill 实际形状 3.82 vs 1.18。结论:FlashInfer 单序列为 Triton 的 1.3-1.8x,chunked 形状 3.2x(其把多序列 batch 进一个 kernel)。`ptx_kda` 单序列快约 20% 但 chunked 形状慢 2.3x,且未带 checkpoint 测量。
- 端到端:Kimi K3 NVFP4 dummy weights,8xGB300 TP8/DCP8,bs1,131k in/1k out,95% prefix hit,A/B/A x10:TTFT 中位数 0.429 s(Triton)→ 0.411 s(FlashInfer);KDA 每层 GPU 时间(EXTEND trace)421 µs → 219 µs。

## 验证与回归风险

新增注册测试 `test_kda_prefill_flashinfer.py`(B200 runner,要求 FlashInfer ≥0.7.0、SM100/SM103),覆盖 bf16/fp32 state、padded/strided 布局、512 前缀的 DCP8 场景、单 token Triton 回退,与 Triton 参考对比(atol/rtol 3e-2、checkpoint 5e-2)。PR 正文给出 GLM-5.3-Flash(4xB300 TP4)与 Kimi K3(8xGB300)的 cold/cached logprob 差异与分支污染检查;B200 rerun 的 GLM-5.3-Flash GSM8K 500 题得分 0.932。风险点:Warm repeat 与对齐边界控制在 GLM 上不 bit-exact,无法声称缓存精确等价;FlashInfer prefill 不支持 TBO/full prefill CUDA graph,拒绝无界 safe gate(原版 Kimi Linear 须用 Triton prefill);依赖树 pin FlashInfer 0.7.0.post1。

## 关联

依赖 FlashInfer #5021(0.7.0 KDA prefill checkpoints);分支缓存污染问题依赖已合并的 #42178 修复并在本 PR 中复测。pack 未列出其他同分类 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
