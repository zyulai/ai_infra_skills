# PR #5432: [perf] introduce deep gemm group_gemm_masked as bmm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Alcanderian |
| 合并 | 2025-04-20 |
| 分类 | Attention / Kernel (tier: major;作用于 DeepSeek MLA absorb 的 w_kc/w_vc 投影) |
| Label | high priority |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,3 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/5432 |

## 一句话结论

DeepSeek MLA absorb 的两个 bmm 改用 deep_gemm masked grouped GEMM + per-token-group 量化,8xH200 decode 提速约 3%。

## 动机

MLA 吸收路径中 q_nope 与 w_kc、attn_output 与 w_vc 的两个 batched GEMM 原走 per-tensor 量化 + `bmm_fp8`。per-token-group 量化 + deep_gemm 的 `grouped_gemm_masked` 通常更快,且避免了 `bmm_fp8` 只支持 per-tensor scale 而必须把 block-wise 量化权重 `block_quant_to_tensor_quant` 重定量(有精度损失)的问题。作者注明 multi-batch 吞吐难以直接对比,因为不同量化方法下 expert 负载差异明显。

## 优化点

新增 Triton kernel `_per_token_group_quant_mla_deep_gemm_masked_fp8`:对 (b, t, k) 输入按 128 一组做 per-token-group FP8 量化,输出对齐到 256(遮蔽 GEMM 的最大 block_m),并在 `t_id == 0` 时把 `masked_m[b]` 写为每批有效 token 数(m),供 `m_grouped_gemm_fp8_fp8_bf16_nt_masked` 使用。权重侧在 `post_load_weights` 中对 fp8 kv_b_proj 且 block size 为 128x128、模型 dtype bf16 的情形,保留 block scale(`w_scale_k`/`w_scale_v`),不再做 `block_quant_to_tensor_quant` 重定量;`forward_absorb` 的 w_kc/w_vc 两个投影改走 masked grouped GEMM,输出按 `expected_m` 截断。整个路径由环境变量 `SGL_ENABLE_JIT_DEEPGEMM_BMM` 门控(默认关,需 SM90)。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_kernel.py: 新增 per-token-group 量化 kernel 与 per_tensor_quant_mla_deep_gemm_masked_fp8 封装;新增 _enable_jit_deepgemm_bmm 开关。
python/sglang/srt/models/deepseek_v2.py: forward_absorb 两处 bmm 换 masked grouped GEMM;post_load_weights 保留 block scale。
python/sglang/test/test_block_fp8.py: 新增量化 kernel 与 masked GEMM 的数值测试。
```

量化 kernel 核心(每组 absmax -> scale -> clamp 到 FP8):

```python
y = tl.load(y_ptr + gid * group_size + cols, ...).to(tl.float32)
_absmax = tl.maximum(tl.max(tl.abs(y)), eps)
y_s = _absmax / fp8_max
y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(...)
if t_id == 0:
    tl.store(masked_m_ptr + b_id, tl.num_programs(0))
```

## 性能声明

8xH200 DeepSeek-V3。kernel 级 profile:1-batch 下原 `bmm_fp8` 占 7.9 us,新路径量化的 2 个 stage 为 1.65+1.14 us 加零初始化 kernel 893 ns(旧 deepgemm bmm 路径为 bmm kernel 5.7 us + 量化 2.7 us);50-batch 下 bmm fp8 6.4 us,新路径量化 2.5+1.7 us + 零初始化 992 ns。`bench_one_batch`(bs=1、input-len 1000、output-len 100、tp8):decode 中位延迟 0.01720 -> 0.01660 s,中位吞吐 58.15 -> 60.24 token/s,总时长 1.887 -> 1.815 s,约 3% decode 提升。对比组均为 `SGL_ENABLE_JIT_DEEPGEMM=1`,差异仅在 `SGL_ENABLE_JIT_DEEPGEMM_BMM`。

## 验证与回归风险

精度:DeepSeek-V3-0304,GSM8K 1319 题(tp8 dp8 dp-attention):bmm 关 5 次 0.935/0.937/0.936/0.933/0.939,开 5 次 0.934/0.935/0.939/0.934/0.937,且每轮快 2-5 s。新增单测:量化 kernel 与原生 per-token-group 参考的 allclose(rtol 0.20,覆盖 B=128、num_tokens 7-16384、D 512/128);masked grouped GEMM 与 torch block bmm 参考的相对平均误差 < 0.0001(M 1-8192、N/K 128/512、B=128)。回归风险:(1) 默认关闭(env gate),仅影响显式开启的 SM90 + bf16 + 128x128 block fp8 DeepSeek 类模型;(2) 输出按 256 对齐,mask 之外行为依赖 deep_gemm 语义;(3) TODO 注明量化 kernel 计划用 CudaC 优化,当前 Triton 版本为过渡实现。

## 关联

同区域(DeepSeek MLA FP8 GEMM):后续 DeepGEMM 集成 PR 系列(如 #14162/#22316 W4A8 DeepEP)与本 PR 共享 `SGL_ENABLE_JIT_DEEPGEMM` 生态;PR #10008 的 batched_gemm_afp4wfp4_pre_quant 是 AMD 侧对应物。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
