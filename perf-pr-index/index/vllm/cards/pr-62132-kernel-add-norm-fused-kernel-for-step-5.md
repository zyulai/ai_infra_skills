# PR #62132: [Kernel]Add norm fused kernel for Step-5 Preview and Muse Spark Open

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-10-17 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ci/build |
| Diff | diffstat 缺失(pack 未记录 +N/-N);素材 diff 收录新增 kernel 与 benchmark 两个文件,另有 9 个文件因篇幅未收入 |
| 链接 | https://github.com/vllm-project/vllm/pull/62132 |

## 一句话结论

为 Step-5 Preview / Muse Spark Open 新增单个 CUDA kernel,把 QK RMSNorm、RoPE、KV cache 写入(及可选 Q FP8 量化)融合为一次 launch。

## 动机

PR 正文极简:Overview 只说明本 PR 是 #61727 与 #61879 系列的一部分,Claims 与 Validation 各节均为空。从 diff 可还原背景:Muse Spark Open 类模型的 attention 预处理原先要跑多条独立路径——weightless QK RMSNorm、GPT-J interleaved RoPE(NoPE 层跳过)、`reshape_and_cache_flash` 的 KV 写入,attention 前每次都重复支付多次 kernel launch 与中间张量往返。

## 优化点

新增 `fused_muse_spark_qknorm_rope_kv_insert` CUDA kernel,对 `[q | k | v | o_gate]` 行一次完成:Q/K 的 RMSNorm(fp32 计算,只 round 一次)、GPT-J 交错 RoPE(`cos_sin_cache == None` 时按 NoPE 层跳过)、K/V 写入 paged KV cache,并可选直接输出 FP8 量化的 Q(`q_fp8_out`),省掉单独的 `scaled_fp8_quant`。实现上每个 head 由一组 lane 持有(每 lane 16 字节向量访存,bf16 head_dim=64 为 8 lane),norm 归约只需 3 次 shuffle;slot→(token, head) 的除法用 host 侧预算好的倒数(FastDiv)替代约 20 条指令的硬件除法;SM90+ 上启用 PDL(`cudaGridDependencySynchronize`/`cudaTriggerProgrammaticLaunchCompletion`)与 qkv projection GEMM 重叠;fp8 KV cache 在 round 到模型 dtype 之后才量化,与 `reshape_and_cache_flash` 的量化边界一致。调度上按 `heads_per_group`(1/2/4)自适应加宽以摊薄寻址开销,仅在能填满一个 CTA wave 时放宽,并对 head_dim=64、(nq,nkv)=(16,2)/(32,4) 的 TP4/TP8 配置按 query head 数设门槛。支持 NVIDIA 与 ROCm。

## 关键改动

```text
csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cu: 新增融合 kernel(norm+RoPE+KV insert+可选 Q FP8),含 PDL、FastDiv、lane-group 归约、fp8 KV cache 路径
benchmarks/kernels/benchmark_muse_spark_qknorm_rope_kv_insert.py: 新增 microbenchmark,对比 fused 与 unfused(fused_qk_norm_rope + reshape_and_cache_flash),eager/CUDA graph 计时,输出 GB/s 与 SOL%
```

其余 9 个未收录文件(pack 注明)无法列出。

```cuda
float sumsq = 0.0f;
#pragma unroll
for (int i = 0; i < kElems; i++) sumsq += elems[i] * elems[i];
sumsq = groupReduceSum<kLanes>(sumsq, group_mask);
float const rms_rcp =
    rsqrtf(sumsq / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
for (int i = 0; i < kElems; i++) elems[i] *= rms_rcp;
```

## 性能声明

PR 正文未提供任何 benchmark 数字(Claims/Validation 空白),也未注明测试条件;仅附带上述 microbenchmark 脚本供复测(其输出含 fused/unfused/tuned 时间与带宽 SOL%,但 PR 内未贴结果)。

## 验证与回归风险

素材 diff 中未见新增单测,PR 正文 Validation 为空,可用回归手段只有新增 benchmark 脚本与 CI。影响面集中在该类模型的 attention 预处理:覆盖 bf16/fp16 输入、fp8 KV cache(E4M3/E5M2)、NoPE 与 RoPE 两种层、paged KV 写入;pre-SM80 上 bf16 模板体直接 return 的守卫保留;fp8 量化边界与 `reshape_and_cache_flash`/`scaled_fp8_quant` 对齐。其余 9 个未收录文件的改动无法评估。

## 关联

同系列:#61727、#61879(本 PR 为其一部分)。同分类其他 PR 素材未给出编号关联。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
