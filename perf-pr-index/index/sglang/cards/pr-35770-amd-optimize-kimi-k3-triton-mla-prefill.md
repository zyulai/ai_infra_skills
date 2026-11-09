# PR #35770: [AMD] Optimize Kimi-K3 Triton MLA prefill on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @clintg6 |
| 合并 | 2026-09-05 |
| 分类 | Attention / Kernel |
| Label | amd, deepseek, run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 extend_attention.py、environ.py、triton_backend.py、新增 split_dim 与测试,另有 6 个文件因超长未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/35770 |

## 一句话结论

为 Kimi-K3 在 gfx950 上重调 Triton MLA prefill:K3 形状双 stage + exp2、新增 4x128 split-dim absorbed kernel、默认开启 FP8 prefill,长上下文 68K/350 吞吐 geomean +16.67%。

## 动机

Kimi-K3 在 MI355X TP8 prefill 用两种注意力表示:fresh/zero-prefix MHA(12 heads,QK/V 192/128)与 cached-prefix absorbed MLA(12 query heads、1 KV head,QK/V 576/512)。现有 Triton prefill 留有性能空间:精确 K3 kernel 只有 1 个 pipeline stage 且用自然指数 online softmax;absorbed 576/512 prefill 一次维护完整 512 宽的 value accumulator,寄存器压力大;FP8 cached-prefix 的 P·V 会让小的 softmax 概率下溢;Aiter 支持原生 FP8 zero-prefix prefill 而 Triton 缺少等价的可选路径。

## 优化点

三块:(1) K3 prefill 调优——gfx950 的 K3 192/128 与 N32 576/512 kernel 用 2 个 pipeline stage;把 log2(e) 折进 attention scale 用 `exp2` 替代 `exp`(LSE 存储时转回自然对数);保留原 compact ragged-grid 调度与 Triton-3.7 fallback;gfx950 head_dim 在 (128,256] 的块尺寸调为 BLOCK_M=128/BLOCK_N=64/8 warps。(2) 新增 4x128 split-dimension absorbed kernel:Q/K latent 4x128、Q/K tail 64、V/output 4x128,单份 online-softmax 状态配四个独立 FP32 value accumulator,归一化后拼接 128 宽输出;dispatch 严格限定 gfx950 + 12 query/1 KV head + 576/512 + BF16 当前 Q/K/V(缓存前缀 BF16 unit scale 或 opt-in FP8 with descales)+ causal + page size 1 + 无 mask/SWA/logit cap/sink/LSE 等,不满足立即走原 generic kernel。(3) FP8 prefill——新开关 `SGLANG_TRITON_FP8_PREFILL_ATTN`(默认开):zero-prefix 192/128 在 launch 前把 BF16 Q/K/V cast 成 FP8(对齐 Aiter 行为);absorbed 576/512 保持当前 chunk BF16、缓存前缀用 FP8;softmax 概率乘 FP8_MAX 后再做 FP8 P·V,accumulator 除回同因子防下溢;支持非 unit 的缓存前缀 K/V descale。另新增 `SGLANG_TRITON_DENSE_PREFILL_ATTN` 与 dense 单循环 prefill kernel(`_fwd_kernel_dense_prefill`):当 batch 适配 chunk 预算时把前缀从 latent KV cache 上投影为 dense per-head K/V(`MHA_ONE_SHOT`),单一指针/单一 dtype 一次 sweep,内部 MASKED=False 消谓词、K/V 载入拓宽为 dwordx4,FP8 用指数偏置(exp2(qk - m + log2(FP8_MAX)))替代乘 FP8_MAX,省一个 BLOCK_MxBLOCK_N 乘法与舍入步。

## 关键改动

```text
python/sglang/kernels/ops/attention/extend_attention.py: exp2/two-stage/FP8 分支与 dense prefill kernel
python/sglang/kernels/ops/attention/extend_attention_split_dim.py: 4x128 split-dim absorbed MLA kernel
python/sglang/srt/environ.py: SGLANG_TRITON_FP8_PREFILL_ATTN 与 SGLANG_TRITON_DENSE_PREFILL_ATTN(均默认 true)
python/sglang/srt/layers/attention/triton_backend.py: dense prefill 路由
test/registered/unit/layers/attention/test_triton_mla_prefill_gfx950.py: 4 个 BF16/FP8 parity 与 dispatch 测试
```

FP8 P·V 的防下溢处理(extend kernel 路径):

```python
if USE_FP8_PREFIX:
    p_dot = (p * FP8_MAX).to(v.dtype)
    acc = acc * re_scale[:, None] + tl.dot(p_dot, v) * (v_scale / FP8_MAX)
```

## 性能声明

内核级(MI355X,device warmup + HIP-graph replay,对未修改 main):zero-prefix BF16 8K 0.726 → 0.415 ms(1.75x);zero-prefix FP8 8K 含 cast 0.420 ms(1.73x);absorbed BF16 8K 前缀+8K extend 8.658 → 4.133 ms(2.09x);absorbed FP8 6.770 → 3.687 ms(1.84x)。FP8 相对优化后 BF16:16K 1.07x、32K 1.22x(8K 无差异,默认跟随 Aiter 全长度启用)。端到端(Kimi-K3 TP8,吞吐 uplift 为并发 2/4/8/16 的 geomean):ISL/OSL 8192/1024 **+1.70%**;ISL/OSL 68K/350 **+16.67%**(并发 16 达 +19.6%)。长上下文收益主要来自 prefill/TTFT,median ITL 两组基本不变。

## 验证与回归风险

数值:BF16 absorbed split 对 main generic 输出 relative L2 约 9.5e-5;FP8 absorbed split 对修正后的 generic FP8 路径约 6.6e-5;zero-prefix FP8 对 BF16 约 3.4-3.5%;模型级 GSM8K(FP8 prefill 开)98%。测试:4 个 K3 BF16/FP8 parity 与 dispatch 测试通过并注册进 `stage-b-test-1-gpu-small-amd-mi35x` suite;generic Triton-3.7 fallback 测试在 gfx950-only gate 后通过;Ruff/Black/Isort 通过。风险点:所有专用路径严格限定 gfx950 与精确 K3 形状,其他平台自动回退,但 `_get_block_sizes_for_extend_attention` 对 head_dim (128,256] 的块尺寸变化影响非 K3 的该区间形状;FP8 prefill 默认开启(8K 有微小 crossover 损失),3.4-3.5% 的 FP8-BF16 相对误差与 98%(非完整)GSM8K 精度依赖后续长上下文 logits 与 workload 验证;dense prefill 会 materialize 整个 batch 的 K/V,受 chunk 预算门控;CI Extra 显示失败标记(x)。

## 关联

同区域优化:同作者同日 PR #33599(Kimi-K3 attn-residual 聚合)、Aiter 的原生 FP8 zero-prefix prefill(行为对齐对象)、`SGLANG_MLA_DECODE_TUNE`(environ.py 相邻条目,gfx950 MLA decode stage-1 调优);`fused_gemm_afp4wfp4_split_cat`(MXFP4 kv_b_proj 与 FP8 cast 融合,dense prefill 路径依赖其输出 dtype 组合)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
