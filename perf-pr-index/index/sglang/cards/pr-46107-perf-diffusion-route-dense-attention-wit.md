# PR #46107: [Perf] diffusion: Route dense attention with a sub-tile side to FA4 on Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-24 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, diffusion |
| Diff | pack 未提供 diffstat（diff 内容可用：2 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/46107 |

## 一句话结论

sm_10x 上 `DYNAMIC_CUDNN_SDPA` 遇到一侧不足 128 行 tile 的 dense attention 改路由到 FA4，该类形状 kernel 快 2-3x，LTX-2 stage-1 attention 每 block 180 -> ~100 us。

## 动机

`DYNAMIC_CUDNN_SDPA` 把 sm_10x 上所有 dense 非 causal attention 发给 cuDNN。当一侧序列短于一个 128 行 tile、另一侧不短时，cuDNN 回退到 `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma` 旧 kernel，FA4 在这些形状上快 2-3x。LTX-2 的音频流只有 126 tokens，audio-prompt、audio-to-video、video-to-audio attention 在每个 block 都命中（LTX-2 stage-1 profile 里每 block 6 次 attention 调用占 4 次）。

## 优化点

`sdpa.py` 的 `_use_cudnn_sdpa`：sm_100 上原来无条件 `return True`，改为 `sorted((query.shape[1], key.shape[1]))` 后 `short_side >= 128 or long_side < 128`——恰好一侧不足 128 tokens 的调用走后端已有的 FA4 fallback 路径，其余保持 cuDNN（两侧都短时两者打平，留在 cuDNN）。两个 kernel 都计算精确 softmax attention，只改浮点舍入顺序。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py: _use_cudnn_sdpa 在 sm_100 加 sub-tile 侧判断
python/sglang/multimodal_gen/test/unit/test_dynamic_cudnn_sdpa_routing.py: 新增路由单测（三组用例）
```

关键逻辑（摘录）：
```python
short_side, long_side = sorted((query.shape[1], key.shape[1]))
return short_side >= 128 or long_side < 128
```

## 性能声明

B300，bf16，无 mask，B=1（B=2 相同），microbench：126x1024 17.6->9.5 us；1536x126 13.2->7.5；126x1536 24.4->11.6（FA 2-3x faster）；6144x126 35.6->11.7（3.0x）；126x6144 84.9->35.3（2.4x）；对照面：192x6144 32.0 vs 55.6（cuDNN 更快，保留 cuDNN）、1536x1536 与 126x126 两者持平。192 tokens 以上 cuDNN 已用 sm_100 kernel 且持平或更快。E2E：LTX-2 stage-1 每 block attention 时间 180 -> 约 100 us（与本 PR 及 prompt-mask PR 合计）；stage-1 step 51.5 -> 48.5 ms（两 PR 合计，synced stage timing）。LTX-2.3 在 batch 2 用相同音频 attention 形状。

## 验证与回归风险

新增单测覆盖三组路由：两侧都满 tile（含 192x6144）留 cuDNN、一侧不足 tile 走 FA4、两侧都短留 cuDNN。准确度：4xB300（2 GPU），LTX-2 两段 768x512x121，CFG parallel + BCG，4 prompts，对 cuDNN 运行 PSNR 16.0/19.5/23.1/14.9 dB；参照带为 `quality="exact"` 对默认 `lossless` 的 PSNR 17.0/20.6/23.1/20.8（LTX-2）与 18.1/19.3/17.6/20.4（LTX-2.3）——40 步 CFG 会把任何舍入差异放大到该量级，故 e2e 只能说明变化落在"舍入级改动"的正常带内，非独立证明数值等价。风险仅限 sm_10x + DYNAMIC_CUDNN_SDPA 的 dense 非 causal 路径。

## 关联

同 stack/同区域：#43373（Ulysses 流水线放行 cuDNN SDPA，同一后端）、#43164、prompt-mask PR（与 #46107 合计贡献 stage-1 收益，编号未在正文给出）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
