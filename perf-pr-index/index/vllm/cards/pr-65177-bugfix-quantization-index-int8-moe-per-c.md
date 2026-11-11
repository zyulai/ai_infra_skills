# PR #65177: [Bugfix][Quantization] Index INT8 MoE per-channel weight scales independently of act quant - #58570

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd(PR authors: @gokay-ai @ahsan-ca @fxmarty-amd,重提 #58570) |
| 合并 | 2026-11-10 |
| 分类 | MoE(major) |
| Label | bug, documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ray, torch.compile, ci/build, multi-modality, tool-calling, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, vllm-ir, mrv2, mistral, dflash, rust, kimi, scheduler, kv-cache-manager, glm, minimax, inkling, cohere, agents, mooncake, pooling, beam, hisparse, recipe, nixl, batch-invariance |
| Diff | +285/-54,20 files(素材未提供 diffstat,按 pack diff 计数) |
| 链接 | https://github.com/vllm-project/vllm/pull/65177 |

## 一句话结论

拆开 INT8 MoE 的 activation 与 weight scale 索引:静态 per-tensor activation + per-channel weight 时 routed expert 的 channel scale 此前被整体忽略、输出错误,新增 `per_out_ch_quant` 修复。

## 动机

`QuarkW8A8Int8MoEMethod` 支持 static per-tensor activation scales 配 per-output-channel weight scales,但 legacy Triton `fused_experts` 路径仍用单一 `per_act_token_quant` 同时决定两种 layout:静态 activation(`per_act_token_quant=False`)也落进 tensor weight-scale 分支,routed experts 的 per-channel scales 被忽略(Fixes #58532),MoE 输出错误。此前无 open PR 覆盖该组合(#57092 是相反混合:per-token acts + per-tensor weights,仍 open),故在原 #58570 基础上小幅修正后重提。

## 优化点

把两个 flag 解耦:非 block W8A8 的 Triton kernel(`fused_moe_kernel` 与 batched `moe_mmk` 系列)新增 `per_out_ch_quant` constexpr 专门索引 W scales,`per_act_token_quant`/`per_channel_quant` 只管 A scales;`int8_w8a8_moe_quant_config`/`make_int8_moe_quant_config` 增加 `per_out_ch_quant` 形参,Quark、compressed-tensors、online INT8 按真实 weight scheme 显式传入;Quark INT8(static 与 dynamic 分支)改为从 activation `QuantKey` 推导 `per_act_token_quant`、从 weight `QuantKey` 推导 `per_out_ch_quant`,替代原先从 `weight_qscheme` 推导 activation 粒度的做法。兼容性:直接 Triton launcher(`invoke_fused_moe_triton_kernel`、`dispatch_fused_moe_kernel`、`fused_experts_impl`、`invoke_moe_batched_triton_kernel`)的 `per_out_ch_quant=None` 默认回落到 `per_channel_quant`,保持 legacy 调用方行为;`Fp8MoEMethod` 显式传 False(per-token act + per-tensor weight 的 `VLLM_BATCH_INVARIANT=1` 场景);factory 默认值本身不变。作者注明这是 stopgap:按 #61024,`FusedMoEQuantConfig` 之后应迁到 `QuantKey` 并删掉这两个 flag。W8A16 路径不需要该 flag(其 kernel 恒按 per-output-channel 索引)。

```python
# Weight-scale layout is independent of activation-scale layout.
else:
    if per_out_ch_quant:
        b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)
    else:
        b_scale = tl.load(b_scale_ptr)
    if per_act_token_quant:
        ...
```

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: fused_moe_kernel 增加 per_out_ch_quant constexpr,W/A scale 索引解耦;launcher/dispatch/fused_experts 链路透传
vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py: moe_mmk 与 batched Triton kernels 同样拆分
vllm/model_executor/layers/fused_moe/experts/triton_moe.py: TritonExperts 传 per_out_ch_quant
vllm/model_executor/layers/fused_moe/config.py + oracle/int8.py: 量化配置工厂新增 per_out_ch_quant 形参
vllm/model_executor/layers/quantization/quark/quark_moe.py: 从 activation/weight QuantKey 分别推导两个 flag
```

## 性能声明

无性能数字。正文明确 "No model eval: this is scale indexing, not a serving/accuracy algorithm change"——这是 scale 索引的正确性修复而非加速,PR 未提供 benchmark。验证平台 MI350X(gfx950)、ROCm nightly 镜像,列出的是单测通过数,非性能数据。

## 验证与回归风险

MI350X 上:`test_quark.py -k int8` 23 passed;`test_moe.py -k launcher` 3 passed;`test_batched_moe` + `test_zen_cpu_int8_moe` 474 passed/509 skipped;`test_triton_moe_ptpc_fp8` + `test_int8_kernel` 76 passed;共享 Triton MoE 内核相关的 `test_moe`/`test_block_int8`/`test_modular_kernel_combinations` 共 858 passed/382 skipped。关键新测试 `test_quark_int8_static_per_channel_weight_scales`(GPU)扰动 routed 与 unrouted expert 的 per-channel scale:main 失败、本 PR 通过。回归风险:factory 调用者若依赖 `per_act_token_quant=True` 隐式启用 per-channel weight scales,现在必须显式传 `per_out_ch_quant=True`,否则静默改为 per-expert tensor scales——5 个 in-tree 测试(test_triton_moe_ptpc_fp8、test_int8_kernel、test_cpu_fused_moe、test_deepep_moe、test_deepep_v2_moe)随此更新;CPU int8 kernel 与 DeepEP 用例在该节点不可用,依赖 CI。

## 关联

Fixes #58532;重提/supersedes #58570;#57092(相反混合 case,仍 open);#61024(后续方向:FusedMoEQuantConfig 迁到 QuantKey)。本批同分类:#51006(MoE router pluggable)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
