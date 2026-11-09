# PR #14105: [LoRA][III] Add LoRA support for MoE layers and enable TP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jonahbernard |
| 合并 | 2026-03-24 |
| 分类 | MoE (tier: major) |
| Label | documentation, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, blackwell, npu, run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);6 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/14105 |

## 一句话结论

让 LoRA 可挂到 MoE 专家层:专用 token 排序 CUDA kernel + 融合 LoRA Triton kernel 把 BAx 增量插进 MoE 前向的激活前与 reduce 前。

## 动机

SGLang 的 LoRA 此前只覆盖 attention 等 dense 层,MoE 专家层(gate_up/down_proj)不支持 LoRA 服务。PR 按 vLLM/HF 的做法把 LoRA 增量融合进 MoE 计算,而不是算完后简单相加,这是初始开发 PR,后续拆分为 #19710 与 #19711 两个小 PR 维护。

## 优化点

四个机制:(1) `FusedMoEWithLoRA` 包装 `FusedMoE`,LoRA 增量插在两个精确位置——base gate_up 投影之后、激活(SiLU/GELU)之前;down 投影之后、跨 top-k 最终 reduce 之前,保证 `y = (W_down + α/r·B_dA_d)·SiLU((W_gate+α/r·B_gA_g)x) ⊙ ((W_up+α/r·B_uA_u)x)` 语义。(2) 新增 `moe_lora_align_block_size` CUDA kernel(jit),先按 LoRA adapter 再按 expert ID 排序 token,产出对齐 block 的 sorted ids。(3) `fused_moe_lora` Triton kernel 以 3D grid `(split-k × M-blocks × N-blocks, len(lora_a_stack), max LoRA adapters)` 一次完成 shrink(A@x)与 expand(B@·)。(4) `TritonRunnerCoreWithLoRA` 复制 Triton runner 并在正确位置插入 LoRA;LoRA 启用时跳过 fused_func(要求非融合路径),并预计算 quant_info。TP 侧提供 `slice_moe_lora_a/b_weights` 按 gate_up/down 模块切分 A/B 权重(gate_up 的 B 按 shard 拼回 packed 布局)。

## 关键改动

```text
python/sglang/srt/lora/layers.py: FusedMoEWithLoRA 包装器与 TP 权重切分逻辑
python/sglang/srt/lora/lora_moe_runners.py: 新文件,LoRAInfo + TritonRunnerCoreWithLoRA(两阶段 LoRA delta 注入)
python/sglang/srt/lora/lora_manager.py: FusedMoE 层识别与 set_lora_info 接线(gate_up_proj_moe/down_proj_moe)
python/sglang/srt/layers/moe/moe_runner/runner.py: MoeRunner 增加 lora_enabled 分支,跳过 fused_func
```

## 性能声明

PR 未提供吞吐/延迟 benchmark,PR 未注明测试条件。提供的 accuracy 数据(Qwen1.5-MoE-A2.7B + LoRA,test_lora_hf_sgl_logprob_diff):60 项 logprob 对比在 1e-1 阈值下 0 FAILED,输出字符串 57/60 与 vLLM 基线一致;logprob 差异 prefill max of max 2.267e+01、mean of max 1.209e+01。

## 验证与回归风险

新增 5 个测试:`test_moe_lora_align_block_size`(排序正确性)、`test_fused_moe_lora_kernel`(BAx 数学正确性)、`test_lora_moe_runner`(单步前向正确性)、`test_lora_moe_vllm_sgl_logprob_diff`(vLLM 缓存基线回归)及 HF logprob diff 的 MoE 用例。风险:仅支持 Triton MoE backend + Triton LoRA backend;`TritonRunnerCoreWithLoRA` 与 `TritonRunnerCore` 手工同步维护(代码重复);shrink/expand block size 硬编码未 autotune(代码内 TODO);正文 Notes 仍写"Only supports tp=1",TP 路径为后续 PR(#19710/#19711)继续完善的对象。

## 关联

正文明确:本 PR 拆分为 #19710、#19711 继续;同区域(卡片库中):#20391(LoRA csgmv kernel offline autotuning)、#24007/#24262(prefill LoRA csgmv)、#34337(multi-adapter LoRA)。
