# PR #46858: [Feat] lfm3: Add text, vision, audio and Omni support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @tugot17 |
| 合并 | 2026-11-16 |
| 分类 | Model-specific Fast Path(major) |
| Label | Multi-modal, run-ci, jit-kernel, run-ci-extra, run-ci-nv-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/46858 |

## 一句话结论

为 LiquidAI LFM3 家族加入 SGLang 原生支持:LFM3(文本)、LFM3-VL(图文/视频)、LFM3-Audio(音频)与 LFM3-Omni(图文音视频混合,含视频音轨)四种输入到文本输出;共享带 gated short convolution 的主干、混合缓存,并新增覆盖宽 tap(K=7)gated causal conv1d 的 CuTe DSL 内核。

## 动机

LFM3 家族此前无 SGLang 支持。其文本主干是带 gated short convolutions、full/sliding-window 混合注意力的线性注意力架构(`conv_L_cache=7`,超出既有 C++ causal_conv1d 的宽 2–4 与共享 Triton 内核的宽 2–5),需要接入 SGLang 的 hybrid cache;四个变体(LiquidAI/LFM3-1.7B、LFM3-VL-2B、LFM3-Audio-2B、LFM3-Omni-2B)复用同一 backbone 与编码器/投影器,可通过 `--trust-remote-code` 用 checkpoint 自带 Transformers 配置与处理器加载。

## 优化点

- 共享 LFM3 主干:gated short conv + full/sliding-window attention + SGLang hybrid cache 集成;CuTe DSL convolution 内核(`cutedsl_gated_conv.py`,含 PyTorch 回退)。
- 新内核设计:token-major (T, dim) 布局免转置;一个线程持一个通道,每 token 读 `proj[t, d..d+31]` 一次 coalesced 事务;K-1 因果窗口存寄存器(无 shared memory、无 barrier);FP32 数学匹配 torch 参考后回写张量 dtype;decode/prefill 双入口,PDL(`use_pdl=True`)与 CUDA graph 兼容,静态 slot 布局把 conv-state slot 偏移折叠为编译期 stride(int64 slot 保 64 位偏移)。
- Vision/Audio 编码器、投影器与 checkpoint 加载;Omni 复用组件与文本主干;多模态预处理用 checkpoint 的 Transformers 处理器与 chat template;Omni 支持混合输入与视频/音频交织(`use_audio_in_video`)。
- `lfm3` 配置组:CUDA 默认 FlashInfer、HIP 默认 aiter;FlashInfer 因 ragged planner 忽略 prefill chunk 内 SWA 而默认 paged prefill;LFM3 暂不支持 speculative decoding 与 DP attention;HIP 无 conv-state checkpoint 的 extra-buffer 交接故关闭。
- transcription 端点适配:`TranscriptionAdapter.build_prompt` 钩子 + `Lfm3AudioAdapter`(decoder-only 采样、支持分块流式 0.5 s chunk)。

## 关键改动

```text
python/sglang/kernels/ops/mamba/cutedsl_gated_conv.py  新增: 宽度泛型 gated causal conv1d (CuTe DSL), decode+prefill
python/sglang/srt/configs/lfm3.py / lfm3_vl.py / lfm3_audio.py / lfm3_omni.py  新增: 配置与 hybrid cache 元数据, 注册 linear-attn model
python/sglang/srt/arg_groups/model_overrides/lfm3.py  新增: attention backend 默认与 FlashInfer paged prefill 强制
python/sglang/srt/entrypoints/openai/transcription_adapters/lfm3_audio.py  新增: 转录 prompt 适配器
python/sglang/srt/entrypoints/openai/serving_transcription.py  text="" → adapter.build_prompt(...)
```

## 性能声明

(全部照抄 PR 正文,未独立核实;PR 正则未检出任何性能数字,本卡不含吞吐/延迟提升声明)

本 PR 是模型接入(Feature)而非优化 PR,正文未提供任何性能基准数字。验证性数据:FP16、TP1、H100、FlashInfer 对 Transformers 的对照——Text 4/4 prefill/decode 检查、4/4 ROUGE-L;Vision 16/16、14/16;Audio→text 5/5、5/5;Omni→text 22/22、20/22。211 个目标 CPU 测试与 259 个子测试全过;Triton 后端 20 个文本/视觉 prefill/decode 检查全过(含超过 256-token 滑窗的长 prompt),生成输出与 FlashInfer 运行全部一致。

## 验证与回归风险

开发 checkpoint 上验证;服务用 stock Transformers 5.19.0 + `--trust-remote-code`。两个 vision ROUGE 失败项在 FlashInfer 与 Triton 下一致存在;Audio/Omni 未用 Triton 测试;另有无 H100 BF16 Omni 冒烟与 H200 BF16 文本/VL(含 TP2 文本)独立验证。风险面:新模型家族 + 新 CuTe DSL 内核 + transcription 端点行为改动(空 prompt 改为 adapter 渲染的 prompt,影响其他 ASR 模型的 chunked 路径语义);`mamba_chunk_size=256` 的取值牵连 extra_buffer 断言与 prefill 缓存粒度;CI 状态三个 run 均为沙漏(pack 摘录时仍在运行)。

## 关联

LiquidAI/LFM3-1.7B、LFM3-VL-2B、LFM3-Audio-2B、LFM3-Omni-2B(checkpoint);`cutedsl_gdn` / `cutedsl_kda`(内核复用其 descriptor 惯用法);LFM2(`Lfm2Config` 的 conv-state 形状与 Mamba2StateShape 复用模式)、NemotronH/FLA(mamba chunk 256 约定)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
