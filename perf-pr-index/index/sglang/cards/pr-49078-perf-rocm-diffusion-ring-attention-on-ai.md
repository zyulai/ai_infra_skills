# PR #49078: [Perf] [ROCM] diffusion: Ring attention on aiter, and reuse the ring's K/V buffers

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/49078 |

## 一句话结论

aiter 返回 ring merge 所需的 softmax LSE 从而加入 ring attention,且 ring 的 stacked K/V staging buffer 改为跨调用复用(顺带修掉 2 GPU 上每 queued layer 一个 ~1 GB pair 的 OOM);8x MI350X 2 GPU 上 MiniMax-H3 较 main Ulysses 2 快 8.4–11.2%。

## 动机

MiniMax-H3 在 MI355X(sglang v0.5.23)2 GPU 上跑 118.2 s,比 1 GPU 多 36% GPU-seconds(B200 上为 6%)。两 GPU 只有一条 XGMI link(~57 GB/s),Ulysses 的 all-to-all 相对层很大:即使有 head-group pipeline(#47567),第一个 group 的输入交换与最后一个 group 的输出交换仍然暴露,main 上 ~46 ms 的层里约占 5 ms。Ring attention 在 degree 2 只发送 K/V 一次、与本地 chunk 的 attention 重叠、且没有输出交换;但 `--ring-degree` 此前拒绝 aiter,而 aiter 是 ROCm 上快的 exact backend。另外 `_ring_attention_varlen` 每次调用新分配一对 stacked K/V:fresh pair 要为 P2P stream 保持到 GPU 到达,host 领先数百层时等于每个 queued layer 一对 ~1 GB,H3 ring 2 在 4 个 denoise steps 内 OOM(`HSA_STATUS_ERROR_OUT_OF_RESOURCES`,standalone loop 亦可复现)。

## 优化点

- aiter 返回 ring merge 需要的 LSE:`forward(..., return_softmax_lse=True)`(dense,`[B, H, S]`)与新增 `forward_ring_kv_chunk`(packed,`[H, Tq]`),并声明 `supports_ring_rotation()`。LSE 为 fp32 自然对数(对照 fp32 reference 校验)。
- Server args(`RING_CAPABLE_ATTENTION_BACKENDS` 加 `"aiter"`)、MiniMax-H3 的 pipeline admission 与 DiT 的 ring gate 接受 aiter。
- `_ring_attention_varlen` 的 stacked K/V pair 改取自 reusable staging buffers(`torch.stack((k, v), out=...)`),不再每次调用分配;此部分 backend 无关、数值不变。
- `_cu_seqlens` 经 `functools.lru_cache` 按 bounds/device 复用上传。文档:ring-capable backend 列表更新。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py: return_softmax_lse / forward_ring_kv_chunk / supports_ring_rotation / _cu_seqlens 缓存
python/sglang/multimodal_gen/runtime/layers/usp.py: _ring_attention_varlen 复用 staging buffer(修 OOM)
python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py: _RING_ATTENTION_BACKENDS = (FA, AITER);ring gate 放行 aiter
python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py: validate_server_args 接受 aiter + ring
python/sglang/multimodal_gen/runtime/server_args/server_args.py: RING_CAPABLE_ATTENTION_BACKENDS 加 "aiter"
docs/docs/sglang-diffusion/parallelism.mdx: ring-capable backend 文档
python/sglang/multimodal_gen/test/unit/test_ring_admission.py: 新增 TestAITerRingKernels(chunks merge == full attention)
python/sglang/multimodal_gen/test/unit/test_minimax_h3_admission.py: aiter ring admission 用例
```

```python
# usp.py: pair 复用,不再每次调用分配
kv_bufs = [
    _a2a_staging_buffer(f"ring_kv_{i}", (2, *k.shape), k.dtype, k.device)
    for i in range(2)
]
torch.stack((k, v), out=kv_bufs[0])

# aiter.py: packed kernel 返回 ring merge 所需的 LSE
def forward_ring_kv_chunk(self, query, key, value):
    """Attend packed [T, H, D] queries to one rotated KV chunk; returns
    the output and the [H, Tq] softmax LSE."""
    output, lse = aiter.flash_attn_varlen_func(
        q=query.contiguous(), k=key.contiguous(), v=value.contiguous(),
        cu_seqlens_q=_cu_seqlens((0, query.shape[0]), query.device),
        cu_seqlens_k=_cu_seqlens((0, key.shape[0]), key.device),
        max_seqlen_q=query.shape[0], max_seqlen_k=key.shape[0],
        softmax_scale=self.softmax_scale, causal=False,
        how_v3_bf16_cvt=AITER_BF16_CVT_MODE, return_lse=True,
    )[:2]
    return output, lse
```

## 性能声明

- 测速环境:MI355X 无空闲容量,故在 8x MI350X(同 gfx950、8-way XGMI mesh;2-GPU all-reduce 56.7 GB/s vs MI355X 的 55.0)上跑;`lmsysorg/sglang:v0.5.23-rocm720-mi35x`,main `ba5906471` 或本分支经 `PYTHONPATH`;recipe 的 flags(aiter、`--performance-mode speed`、eager);H3 T2VA 1344x768、124 frames、49 steps;1 warmup + 3 prompts,client wall clock(0.1 s polling)取 median;四台 2-GPU server 同时跑在 0-1/2-3/4-5/6-7 对上,每臂在同 pair 上对比。
- 结果(2 GPU):v0.5.23 Ulysses 2 = 140.65 s(pair 2-3);main Ulysses 2 = 117.70 s(GPUs 0-1)/ 115.22 s(GPUs 4-5);本 PR ring 2 = 104.51 s(−11.2%)/ 105.52 s(−8.4%)。1 GPU 204 s,故 ring 2 在其一半的 2.5% 以内。
- Ring 只在 Ulysses 只有一条 link 时有帮助:4 GPU 上 Ulysses 4(59.47 s)仍快于 Ulysses 2 × ring 2(66.97 s);8 GPU 上 Ulysses 8(31.22 s)仍快于 Ulysses 4 × ring 2(32.43 s)。
- 其他路线(正文):main 上更多 pipeline groups(4 groups of 7 heads at Ulysses 2)在微基准里每层省 1 ms 但端到端无收益(0.25%);copy engine 的 peer copies 在该机为 28 GB/s(RCCL 的一半),均不能替代 ring。

## 验证与回归风险

- 单元测试(ROCm + aiter):两个 KV chunk 经 ring merge 等于一次全 keys 调用(packed 与 dense 路径均验);aiter under ring 的 admission 测试;新用例在 main 上失败。merge 在 bf16 partial outputs 上工作,与 full 调用差一个 bf16 rounding(atol=rtol=1e-2)。
- 端到端(H3 T2VA,1344x768,124 frames,49 steps):ring 2 输出确定性(跨 run 同一 MP4 md5);对 1-GPU 输出的逐帧 RGB PSNR:ring 2 = 19.5 dB,Ulysses 2 = 18.7 dB;ring 2 对 Ulysses 2 = 20.9 dB;两个不同 prompt 相差 6.6 dB,故 ring 与 Ulysses 属同一跨拓扑浮点重排类别。
- 风险:staging buffer 复用的动机是 2 GPU OOM(4 denoise steps 内),backend 无关且数值不变;ring 仅在 degree-2 单链路场景占优,4/8 GPU 下 Ulysses 更快(正文如实列出);aiter LSE 路径始终走 BF16 kernel(为 ring merging)。

## 关联

#47567(正文提及的 head-group pipeline 前作);无其他明确关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
