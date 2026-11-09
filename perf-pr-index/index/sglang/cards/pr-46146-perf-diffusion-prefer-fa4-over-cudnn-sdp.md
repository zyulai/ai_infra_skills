# PR #46146: [Perf] diffusion: Prefer FA4 over cuDNN SDPA for long self-attention on B300

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-24 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, diffusion |
| Diff | pack diffstat 缺失;diff 为 sdpa.py 的路由条件改动加一个新路由单测 |
| 链接 | https://github.com/sgl-project/sglang/pull/46146 |

## 一句话结论

B300(sm_103)上 ≥16K token、head_dim 128 的自注意力从 cuDNN SDPA 切到 FA4(硬件 exp2),长视频自注意力占 Wan2.2 denoising step 约 67%。

## 动机

`DYNAMIC_CUDNN_SDPA` 把 sm_10x 上所有 dense non-causal 调用发给 cuDNN SDPA——该选择是在 B200 上测得的("cuDNN SDPA measured 1.25-1.5x faster than FA4 CuTe")。但 B300(sm_103)的 FA4 走硬件 exp2 路径与 `tcgen05.ld.red` row max;而长视频自注意力正是 diffusion 模型 step 时间的主体(Wan2.2 A14B 720p 约 67% 的 denoising step)。视频 DiT 在 B300 上整个 denoise 期间顶着 1100 W power cap 运行,因此 kernel 需在持续负载下计时(2 s warm-up 到功耗上限,再 2–6 s 背靠背调用),bf16,B=1 除非另注。

## 优化点

`sdpa.py` 的 `DynamicCudnnSDPAImpl` 增加一条 sm_103 路由:`_prefers_fa4_on_sm103` 在 head_dim 为 128、`Sq == Skv`(自注意力)且 token 数 ≥ 16384 时返回 True,`_use_cudnn_sdpa` 据此提前返回 False 走 FA4。B200(sm_100)、cross-attention、更短序列与 head_dim 64 全部保持 cuDNN——kernel 计时显示 head_dim 64 时 FA4 反而慢 7.7–11.4%。两种后端都是 bf16 + fp32 accumulation 的 exact softmax attention,仅舍入顺序不同(unit-scale 输入 max |diff| 2.4e-4)。与 "Route dense attention with a sub-tile side to FA4 on Blackwell" 触及同一函数,但两条件独立、rebase 无冲突。FA4 的 sm_103 tuning knobs(softmax/correction registers、exp2 emulation、row-max reduction、2-CTA)在 Wan shape 上做过 sweep,默认值即最优。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py: sm_103 长自注意力路由到 FA4
python/sglang/multimodal_gen/test/unit/test_dynamic_cudnn_sdpa_selection.py: 新增路由选择单测
```

```python
_SM103_FA4_MIN_SELF_ATTN_TOKENS = 16384

def _prefers_fa4_on_sm103(query: torch.Tensor, key: torch.Tensor) -> bool:
    return (
        query.shape[-1] == 128
        and query.shape[1] == key.shape[1]
        and query.shape[1] >= _SM103_FA4_MIN_SELF_ATTN_TOKENS
    )
```

## 性能声明

kernel 计时(cuDNN vs FA4,FA4/cuDNN 比值):75600x75600 4x128:7.411 → 7.259 ms(0.979);20x128:36.901 → 36.360 ms(0.985);37800x37800 10x128 0.981;32760 12x128 0.984;16384 24x128 0.993;8192 及以下两者持平(1.003–1.006);37800x512 cross-attn 0.964;18900 24x64:1.077;4096 24x64:1.114(head_dim 64 FA4 更慢)。

Wan2.2 T2V A14B 720p、81 frames、40 steps、4xB300、CFG 2 x Ulysses 2、eager:block attention 38.1 → 36.8 ms;E2E(含 Ulysses pipeline fix)main 92.83 / 92.61 s → 91.23–91.44 s(6 requests,median 91.34 s);在带 NCCL all-to-all 的旧 main 上 `--attention-backend fa` 95.75 s vs 默认选择 97.08 s。注意:pack 检出的 "1.5x faster" 字样出自 B200 上 cuDNN 相对 FA4 的旧测量引用,非本 PR 收益。

## 验证与回归风险

`test_dynamic_cudnn_sdpa_selection.py`:sm_103 将 16K 与 75.6K 自注意力路由到 FA4;8K、512-key cross-attention、head_dim 64 留在 cuDNN;sm_100 在 75.6K 仍 cuDNN。精度:Wan2.2 T2V A14B 720p 4xB300、同 prompts/seeds、x264 crf 0,对比 cuDNN(build 还含其他 bit-identical B300 Wan 改动与一个 63 dB VAE 改动):两 prompts PSNR 27.5 / 22.4 dB,SSIM 0.989 / 0.965;同场景同构图,40-step 轨迹如任何舍入顺序变化一样漂移(最后一帧主体姿态不同)。风险:阈值 16384 与 head_dim 128 硬绑定,其他分辨率/头维度不受益;`_cudnn_failed` fail-safe 钉住逻辑与新路由的交互未在正文讨论。

## 关联

"Route dense attention with a sub-tile side to FA4 on Blackwell"(正文提及,同函数、条件独立,未给出 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
