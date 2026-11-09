# PR #23927: [AMD] Replace fp8 mla with fp8 mha kernel for diffusion model aiter backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-06-09 |
| 分类 | Quantization(major) |
| Label | run-ci, diffusion |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/23927 |

## 一句话结论

扩散模型 AITer 后端的 FP8 prefill 从 MLA ASM kernel 换成 FMHA FP8 ASM kernel,去掉 128->192 pad 与元数据脚手架,Wan 2.2 T2V e2e 1.16x。

## 动机

AITer 后端最初的 FP8 prefill 路径临时借用了 DeepSeek 式 MLA prefill ASM kernel(`mla_prefill_ps_asm_fwd` + `mla_reduce_v1`)在 MI350 系列上为 Wan 2.2 跑 FP8 attention,依赖 ROCm/aiter#2911。该 MLA kernel 有四个硬约束:仅 gfx950 可用、`qk_head_dim=192` 烧死在二进制里(Wan 的 128 每次调用都要零填充)、`v_head_dim=128`、头维按 8 个一组 tiling(`num_heads` 必须被 8 整除,否则 OOB——Ulysses SP=4 下 40 头模型每 rank 10 头会崩溃);还需 `@torch.compiler.disable` 规避 Dynamo。aiter 提供了原生 FP8 FMHA kernel `fmha_fwd_hd128_fp8_gfx950` 后,可以整体替换。

## 优化点

替换为原生 MHA 路径:删除 `_can_use_mla_prefill`、`_build_mla_prefill_metadata`(为 `mla_prefill_ps_asm_fwd` 构造 persistent-scheduling 元数据的整套脚手架)与 `_mla_prefill_ps_attention`(varlen reshape + Q/K 零填充到 192 + kernel + reduce);新增 `_FMHA_FP8_HEAD_DIM = 128` 与 `_can_use_fmha_fp8_prefill`(门控 `_use_aiter_gfx95` + 完整 MHA(`num_kv_heads == num_heads`)+ q/k/v head_dim 均为 128),以及 `_fmha_fp8_prefill_attention`——直接调 `aiter.flash_attn_fp8_pertensor_func`,把 per-tensor descale 转成 shape (1,) 的 float32。由此去掉三样东西:热路径上的 `F.pad`(128->192)、`num_heads % 8 == 0` 约束(解锁 40 头模型的 Ulysses SP=4)、`AITerImpl.forward` 上的 `@torch.compiler.disable`(新路径不再走破坏 Dynamo 的 Python 侧元数据构造,路径变为 torch.compile 友好)。不满足门控时回退 BF16 flash_attn。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py:
  删除 MLA prefill 三件套(kernel 门控/元数据构建/调用封装,约 -220 行)
  新增 _can_use_fmha_fp8_prefill 与 _fmha_fp8_prefill_attention
  (aiter.flash_attn_fp8_pertensor_func,hd128,per-tensor descale)
  移除 @torch.compiler.disable
```

## 性能声明

测试条件:Wan 2.2 T2V,1x MI355,镜像 rocm/sgl-dev:v0.5.10.post1-rocm700-mi35x-20260426,FP8 attention 路径(`SGLANG_DIFFUSION_AITER_FP8_ATTN=1`)。

- Per-kernel:旧路径 `aiter::mla_pfl_qh192_vh128_m32x8_n128x1_causal0` 87,409.1 us + `mla_reduce_v1` 4.4 us = 87,413.5 us;新路径 `fmha_fwd_hd128_fp8_gfx950` 78,006.2 us(单 kernel)。
- End-to-end(一次完整 Wan 2.2 T2V 运行的阶段级 wallclock):Encoding 4.34 -> 4.41 s,Denoising 96.90 -> 81.28 s(-15.62 s),Decoding 11.80 -> 11.79 s;总计 113.04 -> 97.48 s(-15.56 s,-13.8%,约 1.16x faster)。

## 验证与回归风险

精度:PR 正文仅声明"Kernel unit test pass",无数据集级 accuracy 结果;Checklist 全部未勾选,pack diff 未含新增单测。风险:新 kernel 门控为完整 MHA(q/k/v head_dim 均 128、num_kv_heads == num_heads),不满足的形状(如 GQA、非 128 head_dim)自动回退 BF16 路径,行为变化是回退点位置改变;依赖 aiter 版本(ROCm/aiter#2911 合入后的 `flash_attn_fp8_pertensor_func`),旧 aiter 环境可能没有该入口;window_size 硬编码 `(-1, -1, 0)`;影响面限于 multimodal_gen 扩散模型的 AITer 后端 FP8 attention 路径,不涉及 LLM 主干。

## 关联

依赖:ROCm/aiter#2911(FP8 FMHA kernel)。同区域:#21511(AMD FP8 KV cache 与 FP8 attention 使能)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
