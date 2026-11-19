# PR #67302: [ROCm][Quantization] Serve ModelOpt INT8_PB_WO on ROCm with an 8-bit Triton WNA16 kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @AndreasKaratzas |
| 合并 | 2026-11-18 |
| 分类 | Quantization(tier: major) |
| Label | documentation, rocm, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67302 |

## 一句话结论

让 ROCm Triton WNA16 kernel(`triton_w4a16.py`)支持 8-bit `uint8b128` 权重与 group size 16,使 ModelOpt INT8_PB_WO(`quantization="modelopt_int8"`)checkpoint 得以在 ROCm 上加载并服务;8-bit GPTQ 模型同时从 Conch 迁到该 kernel,吞吐 +9-12%,精度不变,4-bit 路径编译 ISA 完全不变。

## 动机

INT8_PB_WO(#65272 引入)今天在 ROCm 的 config 阶段直接拒绝。仅放行 `modelopt_int8` 不够:Exllama 接受 8-bit 权重但只支持 fp16 激活,Conch 只支持 group -1 与 128,所以 bf16 + group 16/32 的 checkpoint 在 ROCm 找不到 kernel;CUDA 上 Humming 已服务这些配置,故新路径按 ROCm 门控。kernel 原来只在 packing 常量里假设 4 bit(每 int32 装 8 个值、`0xF` 掩码、三次 interleave),把位宽提为 constexpr 即可同时支持 4/8 bit,且 4-bit 代码编译出相同指令。

## 优化点

- `triton_w4a16_gemm_kernel` 增加 `BITS: tl.constexpr`,派生 `PACK = 32 // BITS` 与 `MASK = (1 << BITS) - 1`;shifts 改为 `tl.arange(0, PACK) * BITS`;interleave 次数用 `if BITS == 4` 控制第三次;zeros 解包同型处理。
- 支持列表扩为 `TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 16, 32, 64, 128, 256]`,量化类型加 `scalar_types.uint8b128`;`can_implement` 中"N 必须被 8 整除"改为按 pack 整除。
- 加载期 repack(`process_weights_after_loading`)与 qzeros 形状断言全部按 pack 参数化。
- `vllm/platforms/rocm.py` 的支持量化方法列表加入 `"modelopt_int8"`;CUDA 侧行为不变(8-bit 权重与 group 16 是 ROCm-only)。

## 关键改动

```diff
-TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
+TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 16, 32, 64, 128, 256]
 TRITON_W4A16_SUPPORTED_QUANT_TYPES = [
     scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
     scalar_types.uint4,  # asymmetric with explicit zeros
+    scalar_types.uint8b128,  # symmetric int8 (bias=128)
 ]
```

```python
# kernel 内(pack diff 摘录)
    PACK: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    ...
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        if BITS == 4:
            b = tl.interleave(b, b)
        # Extract the correct BITS-wide value for each output column
        b = (b >> shifts) & MASK
```

```diff
 # vllm/platforms/rocm.py
         "modelopt_fp4",
         "modelopt_mxfp8",
         "modelopt_mixed",
+        "modelopt_int8",
         "fp8_per_tensor",
```

## 性能声明

- 8-bit GPTQ(Qwen2.5-1.5B-Instruct-GPTQ-Int8)在 ROCm 从 Conch 换到本 Triton kernel,得到 9-12% 更多吞吐、精度相同(`vllm bench throughput --input-len 512 --output-len 128 --num-prompts 1000`:main 107.6 / 107.7 req/s,本 PR 117.6 / 120.9 req/s;在本分支强制 Conch `VLLM_DISABLED_KERNELS=TritonW4A16LinearKernel` 得 104.8 / 101.0 / 106.7,对 119.3 / 120.1 / 119.5)。
- Kernel 微基准(HIP graphs、冷权重,8-bit group 128):从 M=1 到 8192,所有 Llama-3-8B 与 Qwen2.5-1.5B 线性形状都比 Conch 快(M=1 约 1.1x,大 M 到 6-10x),fp16 下也快于 Exllama。
- 4-bit 路径不变:同 compiled ISA,latency ratio 1.000。
- 精度(gsm8k 5-shot flexible-extract,自量化的 Qwen2.5-1.5B INT8_PB_WO g32,limit 500):bf16 被拒 → 0.554;同权重反量化 dense 对照 0.552;GPTQ-Int8 全量 fp16 0.590(Conch)→ 0.591(Triton)、bf16 0.583 → 0.583。
- 环境与对照:MI300X(gfx942),全部在本分支上对 `origin/main` 778f0fe600 测。

## 验证与回归风险

- 测试:`kernels/quantization/test_triton_w4a16.py`(新增 4/8-bit 参数化 case,含 G=16、G=32、bits=8)、`quantization/test_modelopt.py::test_modelopt_int8_pb_wo_checkpoint` 与 `::test_modelopt_int8_pb_wo_routes_to_its_config` 过 10/10 loops(各 34 tests);checkpoint 测试原先在 ROCm 被 skip、routing 测试在 main 上失败(`modelopt_int8 quantization is currently not supported in rocm`)。
- 无回归:`test_modelopt.py`、`test_auto_gptq.py`、`test_gptq_dynamic.py`、compressed-tensors WNA16 测试、`kernels/quantization -k 'gptq or exllama or conch'` 对照 main 全过。
- 风险(正文明示):gfx950 硬件未覆盖;AMD CI 中跑 `test_modelopt.py` 的 Quantization job 在 main 上禁用直到 #66672 落地;kernel 测试跑 MI355(AMD kernels job)。

## 关联

- INT8_PB_WO 方法由 #65272 加入;#67263 是在 ROCm skip 该 routing 测试的临时手段,与本 PR 互补(本 PR 使该测试在 ROCm 通过);依赖 #66672 恢复 AMD Quantization CI。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
