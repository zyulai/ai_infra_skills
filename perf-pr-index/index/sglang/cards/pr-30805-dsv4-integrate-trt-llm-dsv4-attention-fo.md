# PR #30805: [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @akhilg-nv |
| 合并 | 2026-09-10 |
| 分类 | Attention / Kernel(major) |
| Label | high priority, deepseek, blackwell, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, memory-pool |
| Diff | 素材未注明 diffstat(diff 片段可见 6 个文件,含 2 个新增,另有 8 个文件因篇幅略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/30805 |

## 一句话结论

为 DeepSeek-V4 在 SM100/103 上新增 trtllm-gen 稀疏 MLA 注意力后端,B200 fp8 单卡内核基准 prefill 约 1.2x、decode 约 1.45x 快于 FlashMLA。

## 动机

DSv4 风格注意力(CSA、HCA)此前在 CUDA 上走 FlashMLA 路径;TRT-LLM 的 trtllm-gen 稀疏 MLA 内核在 Blackwell 上有更快实现,值得作为可选后端接入。另外 FlashMLA 变体在 TP>1 时需要把 head_dim pad 到 64/128,而 trtllm-gen 在多卡下扩展性更好。

## 优化点

新增 opt-in 后端 `--dsv4-attn-backend trtllm`(可选 auto/flashmla/trtllm,prefill 与 decode 共用选择):将 SWA/压缩 KV 池切换为统一 512 维 FP8 布局(FlashMLA 为 584 字节 packed 布局),decode、target-verify、draft-extend 与 varlen prefill 全部通过 flashinfer 的 `trtllm_batch_decode_sparse_mla_dsv4` 内核执行;prefill 复用 decode 内核、每 token 一行稀疏表。元数据侧 `init_trtllm_sparse_buffers` 预构建"128 列 SWA + 压缩 KV"合并索引表(仅 c4 尾部与 lens 逐层刷新),行数 pad 到 64-token tile 边界以满足 VarSeq 读取。工程配套:128 MB 共享零工作区;patch FlashInfer 的 multi-CTA 信号量 buffer(按 query 行而非请求数定容)并加容量守卫;独立 `compressor_trtllm.py` 未融合压缩写入管线(融合版留给 PR #32975)。约束:仅 SM100/103、kv_cache_dtype=fp8_e4m3、需 chunked prefill、不支持 hisparse/context parallelism/PD 分离(uniform-FP8 KV 布局未进 PD 握手,见 issue #37838)、KV 固定 per-tensor scale 1.0、不支持在线压缩。

## 关键改动

```text
python/sglang/srt/layers/attention/deepseek_v4_trtllm_backend.py: 新增;DeepseekV4TrtllmAttnBackend(decode/varlen prefill 走 trtllm-gen 稀疏 MLA)+ 多步 spec 后端 + 工厂函数
python/sglang/srt/layers/attention/dsv4/compressor_trtllm.py: 新增;uniform-FP8 池的未融合压缩存储
python/sglang/srt/arg_groups/deepseek_v4_hook.py: trtllm 后端的启动约束校验(SM/kv dtype/CP/PD/chunked prefill)
python/sglang/srt/layers/attention/deepseek_v4_backend.py: DSV4AttnMetadata 增加 trtllm 合并表字段与 init_trtllm_sparse_buffers
python/sglang/srt/arg_groups/fields/exec_.py: 新增 --dsv4-attn-backend 参数
```

```python
out = trtllm_batch_decode_sparse_mla_dsv4(
    query=q_fp8, swa_kv_cache=swa_kv_cache,
    workspace_buffer=self.trtllm_workspace_buffer,
    sparse_indices=sparse_indices,
    compressed_kv_cache=compressed_kv_cache,
    sparse_topk_lens=sparse_topk_lens, seq_lens=seq_lens,
    bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,
    sinks=attn_sink, kv_layout="HND",
)
```

## 性能声明

- 单元内核基准(私有 benchmark 仓库,B200,fp8,TP1,vs `flashmla_dsv4`):prefill(batch 1/2,seq 4096–65536)约 1.2x 快;decode(低/中并发 x 长上下文,及高并发)约 1.45x 快。TP>1 时 trtllm-gen 扩展性更好(flashmla 需 pad head_dim 至 64/128)。
- E2E bench_serving(random 131072 输入/128 输出,8 prompts,并发 4):flashmla 输入吞吐 17500.55 tok/s、mean TTFT 19009.87 ms(median 19636.34)、mean TPOT 86.15 ms、median ITL 12.12 ms;trtllm_gen 输入吞吐 17448.78 tok/s、mean TTFT 16808.71 ms(median 15734.66)、mean TPOT 104.18 ms、median ITL 13.11 ms——即该负载下 TTFT 改善约 12–20%(median 口径),但 TPOT/ITL 略差。
- 精度:DeepSeek-V4-Pro,GSM8K flashmla 0.965 / trtllm_gen 0.985;GPQA(198 题 x 8 repeat)FlashMLA mean 0.885 / TRTLLM mean 0.907。

## 验证与回归风险

验证为 PR 正文中的 GSM8K/GPQA 评测与内核/E2E 基准;checklist 四项(含单测)均未勾选,PR 未附新增单元测试。风险面:默认后端 `auto` 解析仍为 flashmla,行为不变;trtllm 路径以多重 assert 限制启用条件(Blackwell、fp8 KV、chunked prefill、无 CP/PD),但 `DeepseekV4AttnMetadata` 新增字段与 `copy_`/graph-replay 刷新逻辑改动触及 flashmla 路径共用的元数据管理;FlashInfer 信号量 monkey-patch 是 WAR,依赖 FlashInfer 私有 API(`flashinfer.mla._core`),版本升级可能失效;FP8 query 直接 cast(单位 scale)要求 checkpoint k_scale 为 1.0。pack 内 CI:Base/Extra 通过,AMD ROCm 7.2 失败(该 PR 面向 NVIDIA Blackwell)。

## 关联

后续:PR #32975(规划中的融合 uniform-FP8 压缩存储,替换本 PR 的未融合 compressor);issue #37838(PD 握手需携带 KV 布局标识)。同区域:DSv4 注意力/FlashMLA 后端相关 PR(素材未列出编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
