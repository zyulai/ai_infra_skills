# PR #10078: feat: Add FP4 (E2M1) KV Cache Support with Quantization Utilities for MLA

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang(co-authored-by: @yicwang) |
| 合并 | 2025-11-02 |
| 分类 | Quantization(tier: major) |
| Label | high priority, quant, run-ci |
| Diff | diffstat 未提供(pack 素材无增删行数字) |
| 链接 | https://github.com/sgl-project/sglang/pull/10078 |

## 一句话结论

为 MLA KV cache 新增 FP4 (E2M1) 低精度存储,serving 吞吐较 fp8_e4m3 KV cache 高 17.8%。

## 动机

MLA 模型(如 DeepSeek-R1)把 KV cache 量化到 FP8 后,写入要量化、attention 读取要反量化,开销使性能相对 BF16 直接缓存显著下降(BF16 时 trtllm kernel 可直接吃 KV,零 dequant)。引入 FP4 (E2M1) 按 16 元素 block 的低精度缓存,进一步降低显存占用与量化/反量化成本,同时保持与既有 FP16/FP8 工作流向后兼容。

## 优化点

新增 `--kv-cache-dtype fp4_e2m1`:要求 torch 提供 `float4_e2m1fn_x2`(CUDA 12.8+ / PyTorch 2.8.0+,否则警告并回退 auto)。核心是 `KVFP4QuantizeUtil`:`batched_quantize` 把 [B,M,N] 张量按 16 元素 block 重排,取 block max 除以 E2M1_MAX=6.0 得 `ceil(log2(...))` 指数,加 127 存为 uint8 scale;幅值与 7 个 E2M1 边界比较得 3-bit 码加符号位,两个 FP4 值打包进一个 uint8——全程纯 tensor 运算、`torch.compile` 装饰,CUDA Graph 安全;`batched_dequantize` 用位运算解包并查 E2M1 值表 [0,0.5,1,1.5,2,3,4,6] 乘回 `exp2(scale-127)`。`MLATokenToKVPool` 在 fp4 下 kv_buffer 减半(k//2 uint8)并新增 `kv_scale_buffer`(k/16 per token),`set_mla_kv_buffer` 对 nope/rope 分别量化后经 `set_mla_kv_buffer_triton` 与新写的 `set_mla_kv_scale_buffer_triton` 写入,读取路径在 `get_key_buffer` 内 dequant;`ModelRunner.profile_max_num_token` 把 scale buffer 计入 KV 显存预估;trtllm_mla_backend 的 CUDA graph buffer 相应改为 uint8 半尺寸。

## 关键改动

```text
python/sglang/srt/layers/quantization/kvfp4_tensor.py: KVFP4QuantizeUtil 量化/反量化工具(torch.compile)
python/sglang/srt/mem_cache/utils.py: 新文件;MLA kv/scale buffer 的 Triton 读写 kernel(自 memory_pool.py 迁出并新增 scale kernel)
python/sglang/srt/mem_cache/memory_pool.py: MLATokenToKVPool 的 fp4 池布局、set/get 路径
python/sglang/srt/model_executor/model_runner.py: fp4_e2m1 dtype 解析与显存预估修正
python/sglang/srt/server_args.py: --kv-cache-dtype 增加 fp4_e2m1 选项
python/sglang/srt/layers/attention/trtllm_mla_backend.py: CUDA graph buffer 适配 fp4
```

block-wise 量化核心(摘自 `batched_quantize`):

```python
block_max = reshaped.abs().max(dim=-1, keepdim=True).values
scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)
scaled = reshaped / torch.exp2(scale_exp)
sign_bits = (scaled < 0).to(torch.uint8) << 3
magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)
fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)
packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]
```

## 性能声明

B200;server `--model DeepSeek-R1-0528-FP4 --tp-size 4 --moe-runner-backend flashinfer_trtllm --disable-radix-cache`;client `--goodput ttft:5000 tpot:50 --random-input-len 3500 --random-output-len 1500 --max-concurrency 50 --num-prompts 100`:

| KV cache | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Request goodput |
|---|---|---|---|---|---|
| BF16(default) | 1288.17 | 4285.73 | 4853.44 ms | 35.05 ms | 0.46 req/s |
| fp8_e4m3 | 316.05 | 1051.48 | 6093.23 ms | 153.69 ms | 0.00 |
| fp4_e2m1 | 372.28 | 1238.58 | 5611.71 ms | 130.10 ms | 0.00 |

fp4_e2m1 吞吐比 fp8_e4m3 **高 17.8%**(作者归因于 FP4 量化/反量化比 FP8 快);作者同时说明 BF16 baseline 最好,因为 trtllm kernel 直接消费 BF16 KV、无 dequant 开销。

准确性(DeepSeek-R1-0528,KV16/KV8_e4m3/KV4_fp4_e2m1):gsm8k mean_acc 0.9157/0.9154/0.9124(6595 例);aime25 0.5067/0.4934/0.4000(150 例);gpqa_diamond 0.7707/0.7697/0.7273(990 例)。简单数据集近无损,难数据集有退化但作者称在可接受范围。

## 验证与回归风险

验证:新增 `python/sglang/test/test_kvfp4_quant_dequant.py`,参数化 10 个形状(DeepSeekR1-FP4 MLA `[M,1,576]` 与 gpt-oss-20b MHA `[M,8,64]`,M 至 2,868,672),以 MSE/MAE/PSNR/Relative Error 校验重建精度(断言 FP4/FP8 MSE < 1.0)并对比 FP8 vs FP4 的 quant/dequant 耗时。风险:fp4_e2m1 需 CUDA 12.8+/PyTorch 2.8.0+,旧环境静默回退;启动时打印精度下降警告;`get_key_buffer` 在 fp4 下每次读取都执行 dequant(读取路径开销,BS 大时 TPOT 仍显著高于 BF16);当前显式验证仅覆盖 TensorRT-LLM MLA backend 兼容列表与上述测试形状;MHA 支持列为 Future Work。

## 关联

上下文:#10083(points 1-1)。同区域:FP8 KV cache(fp8_e4m3)与 MLA 显存/精度相关改动;同属 Quantization 分类。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
