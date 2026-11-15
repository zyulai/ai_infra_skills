# PR #64534: [Model] Add Mistral Large 4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NickLucche |
| 合并 | 2026-11-13 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, tool-calling, deepseek, cpu, kv-connector, nvidia, quantization, mrv2, mistral, dflash, kv-cache-manager, hisparse, nixl |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64534 |

## 一句话结论

为 vLLM 新增 Mistral Large 4(Mistral-Large-4-1T-A52B 的 FP8 与 NVFP4,model runner V2):sparse MLA + DSA indexer、cross-layer KV sharing、SSMax、DSpark 投机解码、DCP/PCP 长上下文、NIXL P/D、HiSparse host KV offload,并带 NVIDIA(Blackwell/Hopper)与 AMD ROCm(MI350)kernel;正文未提供任何加速数字。

## 动机

为 vLLM 添加 Mistral Large 4 支持,以 open working branch 形式分享(正文原话:该 PR 不打算按原样合并,通用组件另行拆分上游化)。正文给出 recipe 链接,未列性能对比数据。

## 优化点

- Mistral-Large-4-1T-A52B(FP8)与 -NVFP4,跑在 model runner V2 上。
- Sparse MLA with the DSA indexer、cross-layer KV sharing、SSMax。
- DSpark speculative decoding。
- DCP/PCP long context、NIXL P/D、HiSparse host KV offload。
- NVIDIA(Blackwell、Hopper)与 AMD ROCm(MI350 Series)kernel;ROCm 侧含 `Ml4MonoKernel` specialized decode(gfx950/MI355X,`decode_backend=latency_specialized` 的 GB300(SM103)变体在 CI 中注明无对应 agent)与 fused eager ops(indexer q/k norm+RoPE+quant、SSMax scaling、LayerNorm、indexer K cache write,SSMax 与 RoPE 融合等)。

## 关键改动

> pack 的 diff 被截断(113 个文件未展示),模型实现代码(`vllm/models/mistral/` 等)未在 pack 中展示,以下仅覆盖已展示部分。

```text
.buildkite/test_areas/models_basic.yaml: 新增 B200 "Mistral Large 4" step(pytest models/mistral,忽略 amd_specialized_decode)+ MI355 mirror step
tests/models/mistral/mistral_large4/amd_specialized_decode/test_fp8_kv_cache.py(新增): gfx950 上 ML4 specialized decode 在 fp8 latent KV cache 的 GPU 测试
tests/models/mistral/mistral_large4/amd_specialized_decode/test_runtime.py(新增): specialized decode 开关/拒绝条件/layer kind/step routing 的 CPU 测试
tests/models/mistral/test_fused_ops.py(新增): ML4 fused kernels 对照 unfused ops
tests/kernels/attention/test_triton_decode_attention.py: cross-layer view 测试新增 sliding_window 参数(36 = DSpark draft layer 的窗口)
```

```python
kernel = v.op.Ml4MonoKernel(
    W or W0, S, rank=0, npes=1, group=None, topk=TOPK, launches_per_step=1,
    with_indexer=True, index_max_seq=PAGES * PAGE, index_request_width=S,
    index_page_size=PAGE, index_block_table_stride=PAGES,
    index_cache_dtype="fp8", attention_weight="fp8_block128",
    kv_cache_layout="atom", kv_cache_dtype=kv_cache_dtype,
    native_fp4_mfma=True, rope_dtype="fp32", separate_shared=True,
)
```

## 性能声明

- 正文与已展示 diff 未提供任何加速/吞吐数字(regex 检测亦为 none)。
- 能力与约束(来自测试):specialized decode 默认关闭,需 TP4、bf16 或 fp8(e4m3)KV cache、`VLLM_FP8_MOE_REQUANT=mxfp4` + `VLLM_FP8_MOE_REQUANT_ACT=a4w4`,kernel 服务宽度为 (1, 2, 4)(必须等于 vLLM graph capture sizes,否则 FULL-graph replay 永远用不到);拒绝 PP/DP != 1、DCP/PCP、EP、ubatching、LoRA、sleep mode、torch.compile 等配置,启动时列出全部拒绝理由;checkpoint 的 72 层 layer kind 为 ROPE 25 / ROPE_SHARED 28 / NOPE 9 / NOPE_KVS 9(KV-sharing 层不写 cache)。

## 验证与回归风险

- CI:B200 step `pytest -v -s models/mistral --ignore=models/mistral/mistral_large4/amd_specialized_decode`;MI355 mirror 跑整个 `models/mistral`。
- `test_fp8_kv_cache.py`(gfx950/MI355X):一个 TP4 rank shard 的每类 MoE layer 经 `Ml4MonoKernel`(npes=1)单卡跑随机权重、随机分页 cache 与一个 4 行 DSpark verify step(末行 padding slot -1;KV-sharing 层曾越界读源 cache 约 4 GiB 导致 GPU memory fault,此为修复后的回归);断言 fp8 cache 与 bf16 输出一致(KV-sharing 层 exact,其余层按 vLLM fp8 cache writer 存储、饱和于 ±448、无 NaN)。
- `test_runtime.py`(CPU):off-by-default、开关解析、拒绝理由完整、per-step routing(宽度/prefill/ubatch/非 int position 拒绝)、KV cache dtype 与 KV scale 拒绝(fp8 读 unscaled,k_scale != 1.0 拒绝)、latent/index cache dtype 检查。
- `test_fused_ops.py`:fused kernels 对照 unfused——SSMax 与 FlashInfer RoPE 逐 bit(CUDA);ROCm eager RoPE 参考以 fp32 计算、容 1 bf16 ulp;indexer q FP8 码差 ≤ 1 步且受比例上限约束;LayerNorm 允许小比例元素 1 bf16 ulp(统计顺序不同);cache write 存的字节与 indexer insert op 一致,跳过 padded/负 slot。
- 风险:正文自述该分支为工作分支,通用组件拆分上游化中;pack 未含模型实现代码的 diff,本卡不能评估其实现细节。

## 关联

- Recipe:https://recipes.vllm.ai/mistralai/Mistral-Large-4-1T-A52B(正文给出);通用组件另行上游化,正文未列具体 PR 号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
