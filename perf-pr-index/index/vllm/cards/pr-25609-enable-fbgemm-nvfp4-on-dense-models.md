# PR #25609: Enable Fbgemm NVFP4 on Dense models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @samanamp |
| 合并 | 2025-09-25 |
| 分类 | Quantization |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/25609 |

## 一句话结论

为 dense 模型接入 FBGEMM NVFP4 GEMM 后端,B200 上 Llama3.3-70B e2e +5.27%,部分 GEMM shape 内核最高 +60%。

## 动机

vLLM 的 NVFP4 dense GEMM 此前走 vllm native cutlass(或 flashinfer-trtllm/flashinfer-cutlass)路径。FBGEMM 的 `f4f4bf16` NVFP4 GEMM 在部分 GEMM shape 上明显更快,尤其偏好小 batch(作者报告某些 shape 单内核最高提升 60%),但 vLLM 没有接入该后端,也未提供相应的量化内核 benchmark 对照。

## 优化点

新增 `VLLM_USE_FBGEMM` 环境变量(默认关闭,并纳入 compile hash)。`compressed_tensors_w4a4_nvfp4` scheme 的 backend 选择顺序变为:优先 trtllm(`VLLM_USE_TRTLLM_FP4_GEMM`),其次 fbgemm(开启时),再 flashinfer-cutlass,最后 cutlass。fbgemm 后端在权重加载后把 swizzled weight scale reshape 成 `view(-1).view(torch.uint8)` 的布局,apply 时调用 `torch.ops.fbgemm.f4f4bf16(x_fp4, weight_packed, x_blockscale_u8, weight_scale, alpha, use_mx=False)`,激活的 block scale 同样以 uint8 视图传入;activation 量化仍由上游的 fp4 quantize 完成。另在 `benchmarks/kernels/bench_nvfp4_gemm.py` 中加入 `fbgemm-nvfp4` / `fbgemm-nvfp4-noquant` 两个 provider,权重与激活用 fbgemm 的 `triton_scale_nvfp4_quant` 量化,便于与 vllm native cutlass、trtllm 等逐 shape 对比;fbgemm_gpu 未安装时自动跳过。

## 关键改动

```text
vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py: fbgemm backend 选择、scale uint8 布局、f4f4bf16 apply
vllm/envs.py: 新增 VLLM_USE_FBGEMM 开关并纳入 compute_hash
benchmarks/kernels/bench_nvfp4_gemm.py: fbgemm provider(含 triton_scale_nvfp4_quant 量化路径)
```

关键片段(apply 路径):

```python
elif self.backend == "fbgemm":
    out = torch.ops.fbgemm.f4f4bf16(
        x_fp4,
        layer.weight_packed,
        x_blockscale.view(-1).view(torch.uint8),
        layer.weight_scale,
        layer.alpha,
        use_mx=False,
    ).to(output_dtype)
```

## 性能声明

- 端到端:Llama-3.3-70B-Instruct-NVFP4,B200,TP1,random 数据集 128 prompts(input 2048 / output 512,max-concurrency 128,ignore-eos,no prefix caching):Output token throughput 1941.90 → 2044.23 tok/s(+5.27%,作者称"compared to vllm native gemm"的 e2e 提升);TPOT 49.74 → 45.62 ms;Mean TTFT 7916.73 → 8399.35 ms(略有回退);Request throughput 3.79 → 3.99 req/s。
- 单内核:`bench_nvfp4_gemm.py`,FBGEMM kernel 偏好较小 batch,部分 GEMM shape 最高提升 60%(具体 shape 数值以 PR 内截图为准,正文未列文字数据)。

## 验证与回归风险

精度:lm_eval leaderboard 套件(before/after,共约 146000 条 query):BBH acc_norm 0.5457 → 0.5457;MMLU-Pro acc 0.2194 → 0.2194;IFEval prompt strict 0.8762 → 0.8780、inst strict 0.9137 → 0.9149——无精度回退。回归风险:仅影响 NVFP4(compressed-tensors w4a4)scheme;`VLLM_USE_FBGEMM` 默认 False,不开启时路径与原先一致;需要额外安装 `fbgemm-gpu-genai`,缺失时抛 ImportError;fbgemm scale 以 uint8 视图传入,若上游布局变化需同步;TTFT 在该 benchmark 配置下略升,作者未展开解释。

## 关联

同区域:NVFP4 量化路径的 flashinfer-trtllm / flashinfer-cutlass / cutlass backend(同一 scheme 内的替代实现);pack 未列出其他前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
