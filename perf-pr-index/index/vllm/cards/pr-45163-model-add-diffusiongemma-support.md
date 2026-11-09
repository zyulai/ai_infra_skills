# PR #45163: [Model] Add DiffusionGemma Support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-06-12 |
| 分类 | Model-specific Fast Path |
| Label | documentation, performance, new-model, structured-output, speculative-decoding, ready, ci/build, v1, tool-calling, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/45163 |

## 一句话结论

为 block diffusion 语言模型 google/diffusiongemma-26B-A4B-it 提供支持:固定 canvas 迭代去噪、同批混合 causal/非 causal 注意力,复用 spec decode 数据路径。

## 动机

离散扩散 LLM(dLLM)通过在固定长度 canvas 上迭代去噪生成 token,而非自左向右的自回归解码;vLLM 此前的执行模型、注意力后端选择与调度参数均不支持这类模型。目标模型为 google/diffusiongemma-26B-A4B-it(运行指引见 recipes.vllm.ai,镜像 `vllm-openai:gemma-cu130`)。

## 优化点

配置与调度:新增 `DiffusionConfig`(`canvas_length` 决定每步调度的"speculative tokens"数,`max_denoising_steps` 默认读 generation_config.json);`ModelConfig.is_diffusion` 按 HF config 的 `canvas_length` 侦测;`VllmConfig.num_speculative_tokens` 在 diffusion 模式返回 canvas_length——即把 spec decode 数据路径(draft token ids、scheduled spec tokens)重载为 block 生成语义;强制 v2 model runner;diffusion sampler 会物化 `[num_seqs, canvas_length, vocab]` 的 fp32 中间张量,`max_num_seqs` 默认降到 8(>8 单卡 H200 会 OOM)。模型:单个 Gemma4 backbone 双模式运行(encoder 模式 causal 注意力并写 KV cache,decoder 模式双向注意力只读 encoder KV 不写),decoder 独有 self-conditioning MLP,共享 Gemma4 vision tower 支持多模态。注意力:同 batch 内逐请求 causal/非 causal 混合——Triton unified attention 接受 per-seq causal 张量,FA4 用 `dynamic_causal`(内核须以 causal 编译);FlashInfer 不支持混合注意力,被显式报错/通过 `use_non_causal=True` 排除出自动选择。MoE 专家激活新增 `GELU_TANH` 支持(flashinfer cutlass 与 trtllm nvfp4 路径);vllm-flash-attn 外部依赖 bump 到新 commit;`bench serve` 增加 diffusion 指标(denoising steps、canvas positions、committed tokens/throughput)。

## 关键改动

```text
vllm/model_executor/models/diffusion_gemma.py: 模型/ModelState/Sampler(双模式 Gemma4 backbone)
vllm/config/diffusion.py: DiffusionConfig(canvas_length, max_denoising_steps)
vllm/model_executor/models/config.py: DiffusionGemma 配置校验(排除 FlashInfer、max_num_seqs=8)
tests/kernels/attention/test_mixed_causal_attn.py: 混合 causal 批次内核测试(Triton/FA4)
```

## 性能声明

PR 未提供 benchmark 数字(无吞吐/时延对比);`vllm bench serve` 新增 diffusion 专用指标输出(committed throughput、steps per canvas、committed per denoising step 等)供后续测量。

## 验证与回归风险

新增 `tests/kernels/attention/test_mixed_causal_attn.py`:Triton 路径限 SM90/SM100、FA4 路径限 SM90,与参考 paged attention 对拍(atol/rtol 1e-2);模型注册表加入 `google/diffusiongemma-26B-A4B-it`(trust_remote_code)。风险面:vllm-flash-attn 依赖版本变更影响所有 FA 用户;FA4 `dynamic_causal` 要求内核编译期 causal;`max_num_seqs` 默认值的 proxy 检测(≥DEFAULT_MAX_NUM_SEQS 即覆盖为 8)较脆弱;TRITON_ATTN 后端能力表更新(block_table 支持)。

## 关联

同区域:Gemma4 backbone(模型复用);speculative decoding 数据路径(`num_speculative_tokens` 重载);attention selector 的 `use_non_causal` 机制(见 PR #36847 DFlash)在本 PR 中被 DiffusionGemma 复用扩展。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
