# PR #19830: [Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ProExpertProg |
| 合并 | 2025-07-11 |
| 分类 | Quantization(major) |
| Label | performance, rocm, ready, v1 |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,1 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/19830 |

## 一句话结论

把 dense linear 的 FP8 输入量化封装为 `QuantFP8` CustomOp,使 Inductor 可生成/融合 Triton kernel,并打通 RMSNorm/SiluMul 的自动量化融合。

## 动机

FP8 输入量化此前是自由函数直调 `custom_ops.scaled_fp8_quant`,torch.compile 视其为黑盒 CUDA kernel:既无法生成更优 Triton 实现,也无法与同样是 CustomOp 的 RMSNorm、SiluMul 做编译期融合。量化方式(static/dynamic、per-tensor/per-token)散落在各 quant method 的 forward 分支里,传参(`use_per_token_if_dynamic` 等)在调用点临时决定,不利于编译器看到完整意图。

## 优化点

新增 `vllm/model_executor/layers/quantization/input_quant_fp8.py`:注册 CustomOp `quant_fp8` 的 `QuantFP8` 类,在 layer 的 `__init__` 就确定 static/dynamic 与 `GroupShape.PER_TENSOR/PER_TOKEN`,forward_cuda 落到 `ops.scaled_fp8_quant`,forward_native 为纯 torch 实现(可被 Inductor 编译;含 ROCm fnuz 的 FP8 max=224.0 处理与 num_token_padding)。`Fp8LinearOp` 构造参数从 `use_per_token_if_dynamic` 改为 `act_quant_static` + `act_quant_group_shape`,内部持有 QuantFP8,输出 padding 只在"非 cutlass 且非 ROCm 且非 PIECEWISE 编译"时启用。各调用方(fp8、fbgemm_fp8、modelopt、ptpc_fp8、quark、compressed_tensors 两个 scheme)统一迁移。`GroupShape` 从 `compilation/fusion.py` 移到 `quant_utils.py` 以便广泛引用。非 forward 路径(加载后的一次性权重量化)仍走 CUDA kernel。MoE、attention、INT8 留待后续(MoE 调用嵌套在多层自由方法中;attention 的 op 调用被 `unified_attention` custom op 藏在 compile 之外)。

## 关键改动

```text
vllm/model_executor/layers/quantization/input_quant_fp8.py: 新增 QuantFP8 CustomOp(cuda/native 双实现)
vllm/model_executor/layers/quantization/utils/w8a8_utils.py: Fp8LinearOp 持有 QuantFP8 并重设 dispatch 条件
vllm/model_executor/layers/quantization/utils/quant_utils.py: GroupShape 迁入
vllm/model_executor/layers/quantization/fp8.py 等六个 quant method: __init__ 固化量化配置
benchmarks/kernels/bench_per_token_quant_fp8.py: compiled torch vs CUDA per-token 量化基准
```

关键片段(input_quant_fp8.py):

```python
@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    def __init__(self, static, group_shape, num_token_padding=None):
        self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
```

## 性能声明

B200 serving sweep(`redhatai/meta-llama-3.1-8B-Instruct-FP8` 与 FP8-dynamic,并发 1-20):TTFT/ITL/TPOT 中位数上 torch 配置(custom op 全关)多数领先默认 custom-fp8,如 static FP8 TPOT 4.35 vs 4.60 ms(并发 1)、dynamic ITL 4.54 vs 4.53(基本持平);H100 sweep 同向(TPOT dynamic 5.35 vs 5.57、static 4.98 vs 5.36 ms,并发 1)。延迟 sweep(vs custom-all,FP8-dynamic):decode 批 16-256 中 custom-fp8-rms-fused 1.044-1.110、torch 1.066-1.164;mixed 4-128 中 torch 1.079-1.148;prefill 512/2048 torch 1.072/1.091。作者自述 "torch outperforms all other implementations, including custom-fp8(current default on main)"。lm_eval gsm8k 5-shot(limit 100、并发 50):CUDA kernel dynamic 0.76/0.73 vs torch dynamic 0.74/0.72,static 0.74-0.75/0.73,判定均在噪声内。测试条件:模型/配置如上,具体 GPU 型号仅注明 B200 与 H100 sweep。

## 验证与回归风险

测试:`tests/compile/test_silu_mul_quant_fusion.py` 重构为经 `Fp8LinearOp` + `NoOpEliminationPass` 并参数化 cutlass;`tests/compile/test_fusion.py` 的 RMSNorm-quant 融合测试启用 `+quant_fp8`;新增 per-token 量化 bench(与 CUDA 实现对拍 allclose)。风险面:六类 quant method 的量化语义(static×group_shape 组合)在 init 固化,若某模型在 forward 中才决定量化方式(原 TODO 注明 MLA 曾如此)会受影响;`dispatch_w8a8_scaled_mm` 签名收紧;native 路径与 CUDA kernel 的数值差异(文中 padding 注释)需依赖融合测试对拍。

## 关联

同分类 Quantization 的编译融合链:本 PR 是 `+quant_fp8` custom op 的使能者(后续 RMSNorm+quant、SiluMul+quant fusion pass 可作用于它);TODO 引用 issue #20711(MoE 侧 `QuantFP8` 化)。同区域:PR #23036(weight_loader_v2 参数体系)、PR #64417(op dispatch 开销)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
