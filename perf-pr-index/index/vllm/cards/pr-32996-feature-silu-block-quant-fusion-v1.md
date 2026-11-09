# PR #32996: Feature/silu block quant fusion v1

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Monishver11 |
| 合并 | 2026-04-01 |
| 分类 | Quantization |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/32996 |

## 一句话结论

新增 SiLU-Mul 与 FP8 分组(block)量化的融合 CUDA 内核及 torch.compile 匹配 pass,isolated op 提速约 2.4x。

## 动机

FP8-block 量化模型(W8A8 BlockFP8,如 Qwen/DeepSeek 系)每层 MLP 都执行 SiluAndMul 后再对输出做 dynamic per-group FP8 量化;两条路径分离意味着激活写回 HBM 再读一遍、多一次内核 launch。`fuse_act_quant` pass 此前只覆盖 FP8 static 与 NVFP4,FP8 per-group(128/64)仅 ROCm AITER 可用,CUDA 上是空缺(对应 #27847)。

## 优化点

新增内核 `silu_and_mul_per_block_quant`(`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`):每个线程块处理一个 (token, group) 对,线程数 = group_size(64/128,要求 2 的幂);每线程从 `[gate || up]` 输入各读一个元素,寄存器内算 `SiLU(gate) * up`,经 shared memory 折半归约求 group max,thread 0 计算 scale(支持可选 `scale_ub` 上限与最小安全缩放因子)并广播,随后直接量化写出 FP8/INT8 输出与 fp32 scales——全程不落中间激活。支持 `is_scale_transposed` 两种 scale 布局。编译侧在 `act_quant_fusion.py` 注册 `SiluMulBlockQuantPattern`:把 `SiluAndMul + QuantFP8(per-group)` 图模式替换为融合 op,对 `kFp8Dynamic128Sym`/`kFp8Dynamic64Sym` 两个 QuantKey、`is_scale_transposed` x `is_e8m0` x `is_tma_aligned` 全组合注册,仅 CUDA(sm89+;sm100+ 上用 DeepGemm 时 pass 跳过)。`docs/design/fusions.md` 更新 `fuse_act_quant` 支持矩阵。

## 关键改动

```text
csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu: 融合内核(每 block 一个 token×group)
vllm/compilation/passes/fusion/act_quant_fusion.py: SiluMulBlockQuantPattern 模式匹配替换
vllm/_custom_ops.py / csrc/torch_bindings.cpp: silu_and_mul_per_block_quant op 封装与注册
tests/kernels/core/test_fused_silu_mul_block_quant.py: 内核正确性/opcheck 测试(330 项)
benchmarks/fused_kernels/silu_mul_block_quant_benchmark.py: 三实现对比微基准
```

```cuda
// Step 2: Reduce to find group max
shared_max[tid] = fabsf(result);
__syncthreads();
for (int stride = group_size / 2; stride > 0; stride >>= 1) {
  if (tid < stride)
    shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
  __syncthreads();
}
```

## 性能声明

(RTX 4070,CUDA 13.0)

- 内核微基准(`torch.utils.benchmark`,fp16/bf16,GS 64/128):fused 约 129–134us,对比 unfused groupwise(SiluAndMul + `per_token_group_quant_fp8`)约 312–325us、unfused per-tensor fp8 约 270–278us,小/中形状约 2.4x;最大形状 N=2048 x D=14336 为 997.6/971.5us → 503.7us(约 1.9–2.0x)。
- 端到端(`vllm bench serve`,sonnet 数据集 640 prompts、128 RPS,Qwen2.5-0.5B-Instruct FP8-block,RTX 4070 12GB):默认(无 `+silu_and_mul`,等价 main 对 FP8_BLOCK 的行为)output 3600.37 tok/s、mean TPOT 26.35ms;`+silu_and_mul` 且 fusion 关:6719.51 tok/s、TPOT 25.70ms;`+silu_and_mul` 且 fusion 开:8666.61 tok/s、TPOT 14.20ms、mean TTFT 2615→1279ms。即融合相对不融合再提升约 +29% 吞吐、TPOT 约 -45%。作者注明用 0.5B 模型是显存所限,更大模型/更宽 MLP 收益会更明显。
- 精度:lm_eval gsm8k 5-shot x250 样本,开/关 fusion 的 exact_match 差异在误差棒内,无精度退化。

## 验证与回归风险

内核测试 `test_fused_silu_mul_block_quant.py` 330 passed(覆盖 fp8/int8、group 64/128、scale 转置、非整除 hidden 跳过、NaN/Inf 检查、opcheck);编译 pass 测试 `test_silu_mul_quant_fusion.py` 新增 `TestSiluMulBlockQuantModel` 参数化(CUDA only,DeepGemm 支持时 skip),e2e 融合计数测试同步启用 dense 层 `act_quant_fusion`。风险:融合后量化数值与 unfused 路径在容差 5e-2 内对齐(非 bitwise);pattern 覆盖 e8m0/tma_aligned 组合但 DeepGemm 场景主动跳过;仅 CUDA,ROCm 走 AITER 既有融合。

## 关联

#27847(本 PR 面向的 issue/需求)。同区域融合类:#20839(RMSNorm-gated 融合)、#61343(同文件族 layernorm_utils.cuh 的量化数值优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
