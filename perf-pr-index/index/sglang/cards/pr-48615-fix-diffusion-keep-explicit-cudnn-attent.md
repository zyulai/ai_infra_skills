# PR #48615: [Fix] diffusion: Keep explicit cuDNN attention inside LTX breakable CUDA graphs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-11-10 |
| 分类 | CUDA Graph / Compiler（tier: major） |
| Label | run-ci, diffusion |
| Diff | +18/-17, 2 files（pack diffstat 缺失，按 pack diff 文本计数） |
| 链接 | https://github.com/sgl-project/sglang/pull/48615 |

## 一句话结论

把 TORCH_CUDNN_SDPA 加入 LTX breakable CUDA graph 的 dense-capture 允许列表，cudaGraphLaunch 8,670→30，E2E 快 ~4.8–6.2%。

## 动机

LTX 选择 `TORCH_CUDNN_SDPA` 后端时，dense attention 即便所在 layer 选择了 dense capture，也会落在 breakable CUDA graph 之外执行；实测 30 步 LTX-2.3 请求要执行 8,670 次 graph launch。将显式 cuDNN 后端加入现有允许列表后，整网只有一个 graph segment、serving 30 次 launch。

## 优化点

一行运行时改动：multimodal_gen attention layer 的 dense-capture 允许列表在 `TORCH_SDPA`、`DYNAMIC_CUDNN_SDPA` 之外加入 `TORCH_CUDNN_SDPA`。既有的 opt-in、SP=1、no-varlen、无 attention metadata 守卫全部保留；不新增任何 BCG 模型/场景覆盖。配套把 CUDA 回归测试从单一 `SDPABackend` 扩展为 SDPA / CudnnSDPA 两个后端 × Local / USP attention 的组合：回放改变 Q/K/V 张量与改变 mask 的请求，断言精确输出与预期的 capture/break segment 数（修复前显式 cuDNN 会断成 2 个 segment）。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/layer.py: 允许列表加入 TORCH_CUDNN_SDPA（1 行）
python/sglang/multimodal_gen/test/unit/test_ltx2_bcg_coords.py: 回归扩展到两个 SDPA 后端 × Local/USP
```

```python
and self.backend
in (
    AttentionBackendEnum.TORCH_SDPA,
    AttentionBackendEnum.TORCH_CUDNN_SDPA,   # 新增：显式 cuDNN 也可整体捕获
    AttentionBackendEnum.DYNAMIC_CUDNN_SDPA,
)
```

## 性能声明

B300 SXM6 AC，LTX-2.3 native audiovisual pipeline，BF16 lossless，batch/concurrency 1，768×512、17 帧@24FPS、30 steps；16 引擎 128 视频的冻结 cohort（含独立 A/A 引擎）：E2E 均值 main eager（short/long）4.4435/4.5272 s → final explicit cuDNN BCG 4.2295/4.2449 s，配对降 4.815%/6.225%（A/A 噪声 0.098%/0.321%）；main explicit → final explicit 降 1.974%/1.705%（过预登记的 ≥1.5% 门）。main generic SDPA BCG 仍比 final explicit 快 0.322%/0.285%（略超其 A/A 噪声，作者不主张对 best-main 的加速）。Torch Profiler：`cudaGraphLaunch` 8,670→30，D2D copies 9,491→851；三个 BCG 臂均执行 165,477 个 GPU kernel。121 帧正式 cohort 的 explicit-backend 增益未达预登记 1.5% 阈值（证据保留）。17 帧捕获探针中捕获多花 ~0.61 s（1/289 segment → 1/1 segment）。

## 验证与回归风险

扩展的 CUDA 回归在 baseline 上以 2 segment（而非 1）失败，final head 19 tests / 31 subtests 通过；36 个成功输出与 main 在 MP4 字节、每个解码 RGB 帧、全通道 PCM 完全一致；另有一次 2-GPU CFG=2 检查 16/16 通过（120 replays/rank，无 recapture/miss/fallback）。vLLM-Omni 对比未过预登记的音视频质量门（audio PSNR/SSIM 不达标），不主张对 Omni 的性能优势。既有 native batch=2、CFG=1 失败保持不变、不被主张支持。风险面：仅影响 multimodal_gen LTX 显式 cuDNN 后端的 BCG 捕获路径；捕获耗时略增，固定形状 short/long 重放命中缓存。

## 关联

先前工作 #47301（动态 cuDNN 选择）与 LTX dense-capture 相关工作（PR 正文列为 prior art，均未被本 PR 取代）。无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
