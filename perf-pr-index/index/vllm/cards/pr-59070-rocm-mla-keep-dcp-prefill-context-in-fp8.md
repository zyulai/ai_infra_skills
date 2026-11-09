# PR #59070: [ROCm][MLA] Keep DCP prefill context in FP8 through AllGather

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @whx-sjtu |
| 合并 | 2026-10-13 |
| 分类 | Quantization (tier: major) |
| Label | performance, rocm, ready, ci/build |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用,且注明 1 个文件因超长未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/59070 |

## 一句话结论

ROCm MLA DCP prefill 保留 FP8 缓存上下文做 AllGather,通信字节减半并用 AITER 融合 kernel 一步展开成 K/V。

## 动机

PR 之前,ROCm MLA 在 DCP>1 的 prefill 上下文路径是:本地 gather/dequant 成 BF16 后做 BF16 AllGather,再经 reorg、kv_b_proj 投影、K/V 打包等多个独立步骤。通信 payload 是 BF16 宽度,且后续多步分离操作带来额外开销。作者的目标是在保持 FP8 cache 字节、workspace 容量、chunking 与 merge plan 不变的前提下,压缩通信并融合计算。

## 优化点

核心机制改动:每步 metadata 构建一次紧凑 row map(`context_row_indices`,把 request-major 紧凑行映射到 padded rank-major AllGather 布局);`gather_compressed_context` 以 uint8 视图 gather FP8 字节并直接对 FP8 做 AllGather(通信字节对同一 padded 行数减半);随后用新注册的 AITER `rocm_aiter_gather_kv_b_proj` 自定义 op 一步完成 reorg/dequant/kv_b_proj/K/V 打包。`_supports_fused_dcp_prefill_config` 在 init 时缓存 eligibility(要求 fp8/fp8_e4m3 cache、BF16 query、kv_lora_rank=512 等),不支持的格式/维度/投影布局走原路径并转发 K/V concat 回调。所需 kernel 已在 main 锁定的 AITER v0.1.23 中,无需 bump AITER。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_prefill.py: 新文件,row map/FP8 gather/融合展开三函数
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: 新 _context_parallel_compute_prefill_context 融合路径
vllm/_aiter_ops.py: 注册 rocm_aiter_gather_kv_b_proj 自定义 op(含 fake impl 与 mutation 声明)
tests/v1/attention/test_rocm_mla_dcp_context.py: 新增 445 行测试(row map/字节保持/回退/2-GPU 对齐)
```

## 性能声明

端到端 paired run:8 × MI355X,TP8/DCP8,完整 Kimi-K3,FP8 KV,DSpark3(candidate 含 #59069+#59070,未单拆本 PR 贡献)。TTFT 中位数变化:Cold 32K −1.60%,Exact 32K repeat −7.49%,Append 1024 +0.22%,Fork at 12,032 −2.02%,Repeat fork −7.94%,Repeat sibling −8.79%。分段 benchmark(8-GPU gather-to-K/V-expansion,真实 NCCL AllGather,µs):12,198 tokens 289.05→151.89;49,062 778.18→386.80;98,214 1267.18→709.19;196,518 2397.64→1356.37。GSM8K 全集 strict match 1264/1319 (95.8302%) → 1263/1319 (95.7544%)。单对 decode 诊断 TPOT 9.867→9.840 ms,作者不作统计性提速声明。

## 验证与回归风险

新增 `tests/v1/attention/test_rocm_mla_dcp_context.py`(row map 正确性、gather 字节与 workspace 分区保持、13 种不支持配置回退到原路径、2-GPU DCP prefill 与 dense SDPA 对齐 rtol=3e-2),扩展 `test_rocm_aiter_mla_op_registration.py`、`test_rocm_aiter_mla_mtp_split.py`,并加入 `.buildkite/test-amd.yaml`。风险面:仅影响 ROCm MLA DCP prefill;不支持的 cache 格式、e5m2、量化 weight、bias 等自动回退;无 checkpoint/cache 格式变更,可双向回滚。PR 注明后续 main merge `c90a66e3` 尚未做全模型复跑,AMD CI 14463 失败。

## 关联

同区域:#59069(与本 PR 一起测量的配套改动)、#53166(AITER gather 路径,仅 DCP=1)、#56881(NVIDIA 路径 K/V packing)、#52239(NVIDIA symmetric-memory multicast)、#50682(重复性检查覆盖)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
