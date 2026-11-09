# PR #26206: [GDN] Optimize prefill QKV split dispatch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-02 |
| 分类 | Attention / Kernel |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/26206 |

## 一句话结论

用单个 Triton kernel 融合 GDN prefill 的 QKV 拆分,微基准 2.38x,B200 serving 吞吐 +2.7%。

## 动机

profiling 显示 Qwen3.6 GDN prefill 中,QKV 拆分开销显著:通用 FLA 输入守卫用 `torch.split(...).view(...)` 把 post-conv 的 packed mixed QKV 拆成三个独立张量,产生三次独立的 aten::copy_ 内核(共 18.97 ms);同时 chunk_gated_delta_rule_fwd_kernel_h_blockdim64 的默认配置(num_stages=2)也有调优空间。

## 优化点

在 sglang.jit_kernel.triton.gdn_fused_proj 新增 `fused_qkv_split_gdn_prefill_kernel`:mixed_qkv 按每个 token 为 [all_q | all_k | all_v] 布局,一个 launch 内读一行、按偏移把 Q/K/V 分写到三个连续的 [1, T, H, D] FLA prefill 张量,替换三次独立 copy 内核;支持 strided 的 [T, qkv_dim] 视图(经 stride 参数)。gdn_backend 的 forward_extend 在 CUDA 且 qkv_dim ≤ 8192(MAX_FUSED_QKV_SPLIT_DIM)时走融合路径,否则保留原 `torch.split(...).view(...)` 回退。另外把 chunk_h kernel 的单安全配置从 BV=32/warps=4/stages=2 调到 num_stages=4,并新增 SGLANG_GDN_CHUNK_H_BV / NUM_WARPS / NUM_STAGES 环境变量——保持单配置 autotune,因为该内核原地更新 state pool,多配置 autotune 会在 benchmark 时写坏 state pool,env 只允许本地验证后固定。

## 关键改动

```text
python/sglang/jit_kernel/triton/gdn_fused_proj.py: 新增 fused_qkv_split_gdn_prefill_kernel(单 launch 拆分 QKV)
python/sglang/srt/layers/attention/linear/gdn_backend.py: forward_extend 改走融合 split(qkv_dim<=8192,带回退)
python/sglang/srt/layers/attention/fla/chunk_delta_h.py: chunk_h 配置调至 BV=32/w4/s4,env 可覆盖
benchmark/bench_linear_attention/bench_gdn_qkv_split.py: 连续/strided 输入微基准(含逐元素精确校验)
```

## 性能声明

Serving:Qwen/Qwen3.6-35B-A3B-FP8、单卡 NVIDIA B200、默认 SOTA serving workload(80 随机 prompts,胜出 lane 加 --disable-radix-cache)。Chat 1000/1000:SGLang baseline best 7205.38 out tok/s、TTFT 858.08 ms、TPOT 10.20 ms;patched 7396.73 / 709.25 / 10.07;vLLM latest best 6480.19 / 1975.60 / 10.33。Long 8000/1000:baseline 4243.72 / 4014.37 / 14.55;patched 4353.39 / 3820.19 / 14.28;vLLM 3993.22 / 4275.92 / 15.35。TensorRT-LLM latest 因不识别 model_type=qwen3_5_moe 无法服务。Kernel profile:chunk_h 内核 269.00 us 到 218.78 us(90 次调用);FLA/GDN prep copy 18.97 ms 被移除,换为 fused split 3.33 ms(37.00 us/次)。微基准(H200、seq_len=8192、qkv_dim=6144、bf16):contiguous 124.54 us 到 52.41 us(2.38x);strided 349.45 us 到 338.48 us(1.03x)。

## 验证与回归风险

准确性:MMLU full 0.436761 到 0.437687(+0.000926),GSM8K full(5-shot)0.954338 到 0.955860(+0.001522),同一 1x B200 配置,enable_thinking=false。验证含 py_compile、微基准(逐元素 assert_close,rtol=0)、实机 serving benchmark、对 vLLM 的 torch profiler 对比。回归面:融合路径仅限 CUDA 且 qkv_dim ≤ 8192,其他平台/超大维度走原路径;chunk_h 配置改动影响所有使用 FLA chunk_delta_h 的 GDN/线性注意力模型,env 变量保留验证通道。

## 关联

同区域:Qwen3 Next/GDN 系列优化(如 PR #26204 同作者同模型栈的 MoE 优化);gdn_fused_proj 中已有的 fused_qkvzba_split_reshape_cat 为同文件前序融合工作。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
