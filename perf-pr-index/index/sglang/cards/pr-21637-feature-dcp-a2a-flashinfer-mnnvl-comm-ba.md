# PR #21637: [Feature] DCP: A2A + FlashInfer-MNNVL comm backends and q-replicate (Helix)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @thanhhao98 |
| 合并 | 2026-07-24 |
| 分类 | MoE |
| Label | deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计);diff 有省略(2 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/21637 |

## 一句话结论

为 DeepSeek MLA decode 上下文并行(DCP)新增 a2a/fi_a2a 通信后端与 q-replicate,减少每层集合通信。

## 动机

在已合并的 DCP 基础(#14194)之上,decode 上下文并行的注意力规约此前只有 ag_rs(AllGather + ReduceScatter)一条路径,且每层都要对 Q 做一次 head-dim all-gather。该 PR 引入可插拔通信后端与 q-replicate,降低每层的集合通信次数,并为 GB200 MNNVL 硬件接入 FlashInfer 专用 all-to-all 内核。

## 优化点

新增 `--dcp-comm-backend {ag_rs, a2a, fi_a2a}` 与 `--dcp-replicate-q-proj` 两组开关。a2a 后端把各 DCP rank 的 partial attention 输出与 fp32 LSE 打包进一个张量,用一次 NCCL All-to-All 交换,再用 one-shot Triton LSE combine(base-2)本地合并;配套给 pynccl 增加了 CUDA-graph-capturable 的 `all_to_all_single`(ncclGroupStart/End 融合收发),fp8 KV cache 经 uint8 字节视图传输。fi_a2a 后端只把跨 rank 交换委托给 FlashInfer MNNVL 的 `decode_cp_a2a_alltoall` 内核(GB200),本地 combine 复用同一 Triton kernel,硬件上要求 MNNVL fabric。q-replicate(Helix)在 pre-capture 阶段把 q_b_proj/w_kc 的全 head 权重 gather 一次,之后每层本地直接投影 full-head Q,跳过每层的 Q all-gather,用少量冗余 GEMM 换掉一次集合通信。

## 关键改动

```text
python/sglang/srt/layers/dcp/comm.py: a2a/fi_a2a 后端核心(输出+LSE 打包、MNNVL workspace 初始化)
python/sglang/kernels/ops/attention/dcp_kernels.py: 新增 Triton LSE-weighted combine kernel(base-e/base-2)
python/sglang/srt/distributed/device_communicators/pynccl.py: 新增可被 CUDA graph 捕获的 all_to_all_single
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py: q_replicate_active 路径,跳过 Q all-gather
python/sglang/srt/model_executor/model_runner.py: _prepare_replicated_q_proj,capture 前 gather 全 head 权重
```

## 性能声明

PR 正文 "Results — DeepSeek-V2-Lite (MLA, BF16)" 一节仅标注 "inprogress",未给出任何具体性能数字;素材中也没有可检测到的 perf claim。

## 验证与回归风险

新增三个注册 CI 测试:`test/registered/dcp/test_dcp_fi_a2a_reshape.py`(fi_a2a pack/unpack 布局须与 NCCL a2a 及 CPU 参考一致,作者视为 fi_a2a 的主要正确性风险)、`test/registered/kernels/test_dcp_lse_combine.py`(Triton combine vs CPU 参考,含 NaN/inf 与 cuda-graph buffer)、`test/registered/unit/server_args/test_dcp_config.py`。回归面:q-replicate 仅支持 unquantized bf16/fp16 absorb 路径,量化 q-proj 或非 16-bit w_kc 自动回退原 all-gather;a2a 要求 num_heads 可被 dcp_size 整除;fi_a2a 要求 SM90+ 且 MNNVL fabric(如 GB200 NVL72);activation.py 中 jit act-and-mul 对未对齐 hidden size 回退 sgl_kernel 实现。

## 关联

前序 PR #14194(DeepSeek V2 DCP 基础实现)。同属 DeepSeek MLA decode 上下文并行区域。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
