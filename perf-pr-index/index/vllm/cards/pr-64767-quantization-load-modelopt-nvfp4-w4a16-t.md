# PR #64767: [Quantization] Load ModelOpt NVFP4 W4A16 token embeddings in mixed-precision checkpoints

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @puririshi98 |
| 合并 | 2026-11-11 |
| 分类 | Quantization(tier: major) |
| Label | documentation, ready, ci/build, cpu, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64767 |

## 一句话结论

新增 `ModelOptNvFp4EmbeddingMethod`:ModelOpt mixed-precision checkpoint 里标记 `W4A16_NVFP4` 的 token embedding 可加载并服务——权重表保持 packed、只对查到的行在单个 Triton kernel 里 dequantize;权重 23.05→22.33 GiB、KV cache 增 1.9%,MMLU-Pro / TTFT / ITL 无变化。

## 动机

ModelOpt mixed-precision checkpoint 可在 `quantized_layers` 中把 token embedding 标为 `W4A16_NVFP4`,但此前只有 Q8_0 embedding 能加载(经 b12x)。在 main 上,`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`(BF16 embedding 重量化为 NVFP4)加载失败:`The size of tensor a (4096) must match the size of tensor b (2048) at non-singleton dimension 1`。Root cause:`ModelOptMixedPrecisionConfig.get_quant_method` 只对 b12x block codecs 返回 embedding method,且该方法只接受 Q8_0。

## 优化点

1. `ModelOptNvFp4EmbeddingMethod`(W4A16 NVFP4 token embedding):保留 packed 表(two e2m1 values per byte,low nibble first;每 16 值一个 e4m3 scale;per-tensor fp32 scale),只在 lookup 时对查到的行 dequantize——GPU 上是单 Triton kernel(改编自 `compressed_tensors_embedding.py` 的 INT embedding kernel),CPU 上走 torch 路径。
2. Triton kernel 手工解码 e2m1 与 e4m3fn(不经 fp8 硬件路径,pre-fp8 GPU 也能跑);block size 只由 hidden size 决定,故 `torch.compile(fullgraph=True)` 的 lookup 在动态 token 维度下无需重编译——对比 #42791 的 `dequantize_to_dtype(..., swizzle=False)` 走 `_triton_dequantize_nvfp4`、launch config 依 row count 分支,Dynamo 会 specialize `input_ids.size(0)` 导致启动失败。
3. **拒绝 embedding 与 LM head tie**(LM head 会读到 packed bytes):`get_quant_method` 在 `tie_word_embeddings` 时直接 ValueError,`apply()` 也 raise NotImplementedError。
4. Weight reload 保持 lookup table 地址不变(`refill_derived` 就地填充),captured CUDA graphs 保持有效;`supports_pre_processed_weights = True` 支持预处理权重/fast-start 加载。
5. 模型必须以 checkpoint prefix 构建 embedding(如 Nemotron-H 的做法);EAGLE draft 靠比较 `weight.shape[-1]`(quantized embedding 的 packed 宽度)决定不共享 target embedding,MTP draft 不受影响——此处均未改动。

## 关键改动

```text
vllm/model_executor/layers/quantization/modelopt.py(新增): _nvfp4_embedding_kernel(Triton)+ ModelOptNvFp4EmbeddingMethod + W4A16_NVFP4 分派与 tie 拒绝
tests/quantization/test_modelopt_nvfp4_embedding.py(新增): Triton/torch 零容差对照、tie 错误、reload 地址稳定性
docs/features/quantization/modelopt.md: W4A16_NVFP4 embedding 行为说明
.buildkite/hardware_tests/cpu.yaml: CPU CI job 加入该测试
```

```python
    byte = tl.load(packed_ptr + tid * packed_stride + col // 2, mask=mask, other=0)
    code = (byte.to(tl.int32) >> ((col % 2) * 4)) & 0xF
    # e2m1: exponent bits 2:1, mantissa bit 0, bias 1; exponent 0 is subnormal.
    exp = (code >> 1) & 0x3
    man = (code & 0x1).to(tl.float32)
    pow2 = ((exp + 126) << 23).to(tl.float32, bitcast=True)
    value = tl.where(exp == 0, man * 0.5, (1.0 + man * 0.5) * pow2)
    value = tl.where(code >= 8, -value, value)
```

## 性能声明

(perf claim 正则捕获到 "faster: on RTX 6000D with a 131,072 ×" 一条)

- **加载与显存**:NVFP4 embedding 可加载并服务,权重 23.05 → 22.33 GiB,KV cache 增长 1.9%(5,701,632 → 5,812,539 tokens)。
- **精度与速度不变**:MMLU-Pro 57.90 vs 57.19(2,100 questions,噪声范围内),TTFT 与 ITL 相同。
- **Triton lookup 与 torch 路径 bitwise 相等且更快**:RTX 6000D、131,072 × 4,096 表,1–256 tokens 时 23 vs 81 µs,32K tokens 时 277 vs 5,434 µs(peak scratch 256 vs 2,272 MiB);以动态 token 维度编译。
- **E2E(Triton kernel 之前)**:一台 RTX 6000D(SM120),main `46f36fa294` + 本 PR 与 #64766(无 #64766 时 main 的 b12x linears 在第一个长 prompt 上杀死 engine);TTFT/ITL 为 C=1 下的 ms 中位数:

| | BF16 embedding | NVFP4 embedding |
|---|---|---|
| weights | 23.05 GiB | 22.33 GiB |
| KV cache | 5,701,632 tokens | 5,812,539 tokens |
| MMLU-Pro (2,100 q) | 57.19 | 57.90 |
| TTFT 768 / 2K / 8K, ITL | 136.2 / 314.4 / 1231.7, 6.69 | 136.3 / 314.4 / 1232.4, 6.68 |

## 验证与回归风险

- Unit:`pytest tests/quantization/test_modelopt_nvfp4_embedding.py` 在 RTX 6000D 上 6 passed(main `75be5c3919` + 本 PR);CPU quantization CI job 跑其中 2 个 CPU 测试。4 个 GPU 测试以零容差对照 Triton 与 torch 路径(bf16/fp16,hidden 64 与 2,080,覆盖每个有限 e4m3 scale code 含 subnormal);经 `ModelOptMixedPrecisionConfig` 构建真实 `VocabParallelEmbedding`、走 `weight_loader` 加载,并与 `break_fp4_bytes` dequantization 零容差对照;同时检查两类 tie 错误,以及 rerun post-processing(weight reload 或预处理加载会触发)保持 lookup table 地址。main 上该测试为 `ImportError: cannot import name 'ModelOptNvFp4EmbeddingMethod'`。
- 未运行:GPU 上的 TP > 1(两 rank CPU run 与 reference 完全一致)。
- 回归风险:tie 的 embedding 被拒绝(如 OPT);EAGLE draft 因 packed 宽度比较保持自己的 embedding(行为未变);hidden size 必须能被 16 整除。

## 关联

- #64766(b12x linears,E2E run 的前置)、#42791(更早的 NVFP4 weight-only embedding 尝试,torch.compile 重编译问题)。
- 模型侧要求:Nemotron-H 式的 checkpoint prefix 构建 embedding。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
