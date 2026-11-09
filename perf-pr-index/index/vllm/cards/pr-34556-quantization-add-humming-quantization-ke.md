# PR #34556: [Quantization] add humming quantization kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2026-04-24 |
| 分类 | Quantization(tier: major) |
| Label | performance, ready, gpt-oss, nvidia, quantization |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 4 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/34556 |

## 一句话结论

把 Humming JIT 量化 kernel 库接入 vLLM:支持 W1-W8 x A16/A8/A4 的灵活位宽组合(GPTQ/AWQ/FP8/MXFP4/NVFP4/BITNET 权重),linear 与 fused MoE 双路径,Turing 到 Hopper 全支持且自称性能优于 Marlin。

## 动机

Marlin 系 kernel 固定 4-bit 位宽、编译期特化,位宽组合与量化类型覆盖有限。Humming(inclusionAI 开源)是高度灵活的 JIT 量化 kernel 库:支持 W{1-8}A{16/8/4} 的任意组合,兼容 GPTQ/AWQ/FP8/MXFP4/NVFP4/BITNET 权重类型(不支持 desc_act=True 的 GPTQ 与 HQQ),在 Turing/Ampere/Ada Lovelace/Hopper 上性能最优,并支持加载时在线量化以省显存。接入后 vLLM 用户可用 `--quantization humming` 在更细的位宽点上取性能/显存/精度折中(实验性特性)。

## 优化点

机制上:(1) MoE 侧新增 `fused_humming_moe.py`,以 modular kernel 框架的 `FusedMoEExpertsModular` 实现三个 experts 类:`HummingIndexedExperts`(indexed gemm,标准 contiguous 格式,`moe_align_block_size` 后直接按 expert 索引计算,`moe_fused_mul_sum` 完成 topk 加权归约)、`HummingGroupedExperts`(grouped contiguous,`moe_permute/unpermute`)与 `BatchedHummingGroupedExperts`(grouped masked,BatchedExperts 格式,服务 EP dispatch 场景);gemm 类型经 `VLLM_HUMMING_MOE_GEMM_TYPE` 指定或按 SM90+ TMA 能力自动选 grouped/indexed,workspace 按 buffer 元信息字节量配平两块 workspace。(2) 新增 Triton kernel `moe_fused_mul_sum`:把 expert 输出乘 topk_weight 求和融合为单 kernel,带分架构启发式 BLOCK_M/BLOCK_K/warps 配置,支持 expert_map(EP 无效对跳过)。(3) Linear 侧注册 `HummingLinearMethod`,`LinearBase` 贯通 bias 参数,`weight_loader` 的 packed shard 计算改为 round。(4) 三个环境变量:`VLLM_HUMMING_ONLINE_QUANT_CONFIG`(加载时在线量化,支持 INT1-8/FP3-8、per-layer dynamic 覆盖、force_requant)、`VLLM_HUMMING_INPUT_QUANT_CONFIG`(激活量化)、`VLLM_HUMMING_USE_F16_ACCUM`(Turing/Ada 上 FP16 accumulator MMA 提速,有溢出风险);最小算力要求 SM75。(5) `quantization="humming"` 在 override 顺序中置顶,保证用户显式指定时总是命中。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_humming_moe.py: Humming Indexed/Grouped/Batched experts(mk 框架)
vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py: Triton fused topk 加权求和 kernel(新增)
vllm/model_executor/layers/quantization/__init__.py / humming(HummingConfig,部分在截断 diff 中): 注册 humming 量化方法
vllm/envs.py: 4 个 VLLM_HUMMING_* 环境变量
vllm/config/model.py: humming 加入支持列表并强制 override 优先
```

关键片段(gemm 类型自动选择):

```python
if current_platform.has_device_capability(90):
    # for device that supports TMA, use grouped gemm
    gemm_type = "grouped"
else:
    gemm_type = "indexed"
```

## 性能声明

PR 正文给出 H20/H200 的 kernel benchmark(TFLOPS,详见 humming 仓库 benchmarks/results):

- H20,dense n8192 k8192 channelwise,fp16:humming-w4a16 对 marlin-w4a16 在 m=1 为 4.65 vs 4.28,m=16384 为 142.26 vs 89.65;humming-w8a16 对 marlin-w8a16 为 3.64 vs 3.21(m=1)、141.97 vs 88.41(m=8192);W1-W7 位宽递减平滑(如 m=16384 时 w1a16 142.05 → w7a16 141.14,torch-w16a16 为 138.05)。
- H20,dense fp8(float8e4m3):m=32 处 humming-w8a8 107.46 vs cutlass-w8a8 88.25;m=16384 为 281.27 vs 278.90(m=1 处 cutlass 3.99 略优于 humming 3.88)。
- H20,MoE n512 k7168 e256 topk8 blockwise 128x128 fp8:humming-w8a8 对 triton-w8a8 在 m=1 为 2.56 vs 1.04,m=16384 为 235.64 vs 232.81;w1a8 达 241.87。MoE n7168 k256:m=2048 为 86.32 vs 52.00,m=16384 为 128.20 vs 64.59。
- H200,dense fp16:m=32 humming-w4a16 141.64 vs marlin 128.14;m=512 583.76 vs 397.81(torch-w16a16 峰值 692.02);H200 dense fp8 大 m 下 cutlass 更优(m=16384 cutlass 1375.15 vs humming-w8a8 1072.73),小中 m(如 m=8192 以下)humming-w4a8 等低比特配置有优势区间。

PR 注明 Blackwell(SM100)不支持 UMMA、SM120 无 blockscale 支持性能欠佳;`VLLM_HUMMING_USE_F16_ACCUM` 在 Turing/Ada 上可因 FP32 accumulator MMA 只有 FP16 一半 TFLOPS 而提速,但增加溢出/下溢风险。以上均为 kernel 级 microbenchmark,非端到端 serving 数据。

## 验证与回归风险

依赖 `humming` 库(dtypes/config/layer 接口)与 JIT 编译;`_supports_current_device` 要求 CUDA 且 SM >= 7.5;支持 SILU/GELU/SWIGLUOAI 等激活回调与 EP expert_map。回归风险:`linear.py` 的 `LinearBase.__init__` 增加 bias 形参并在全部并行 Linear 子类透传,属公共接口变更;packed shard 计算从整除改为 round,影响其他 packed 量化方法的 weight_loader 行为;fused MoE 的 weight_loader 中 humming 按 `weight_schema.quant_method` 代理 gpt_oss_mxfp4 分支;在线量化未校准,低比特(6-bit 以下)显著影响精度(PR 建议线上量化用 6-bit 以上)。

## 关联

上游:Humming 库(https://github.com/inclusionAI/humming/)。同为替代/补充 Marlin 路径:索引内卡片 #14447(WNA16 Marlin kernel)、#2497(GPTQ Marlin)、#12978(StreamK block-quantized CUTLASS)、#16850(dense Marlin 优化)、#25609(FBGEMM NVFP4)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
