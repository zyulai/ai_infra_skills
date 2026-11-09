# PR #30575: [AMD] Enable Fast Triton Sparse MLA backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @clintg6 |
| 合并 | 2026-09-10 |
| 分类 | Attention / Kernel (tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(9 个文件未收入,仅含 triton_sparse_mla.py 的完整改写) |
| 链接 | https://github.com/sgl-project/sglang/pull/30575 |

## 一句话结论

为 ROCm fp8 DSA 路径新增纯 Triton 稀疏 MLA prefill/decode 后端并替换旧的 env-var 门控 prefill,隔离 prefill microbench 提速约 10%,端到端吞吐提升 3–5%。

## 动机

此前的 Triton DSA prefill 只能通过环境变量 `SGLANG_DSA_TRITON_PREFILL` 开启,且没有对应的 Triton decode kernel。TileLang 的 partial+combine 实现在 prefill 场景(n_groups=1,attention tile 只有 M=16 heads,相当于一个 16x16 MFMA)存在 256 线程块内的协调开销,需要更贴合小 tile 的 kernel;decode 侧也缺少能在短序列(MTP verify/draft seq=1-6)下填满 GPU 的实现。

## 优化点

核心是重写 Triton 稀疏 MLA kernel 并正式成为显式后端选项:prefill 采用 split-dim 模式,把 512 宽的 nope 维度按 4x128 分组处理,对齐 CDNA 原生 MFMA tile,再加 64 宽 rope tail;长序列走 grid=(seq,) 的单遍 kernel,短序列走 adaptive split-K 路径(grid=(seq, head_blocks, kv_splits) + log-space reduce),`kv_splits == 1` 时合并为单遍 fused 快路径。prefill 还可把 topk 工作拆分到多个 CTA 并在 Triton 内归约 partial online-softmax 状态。所有 kernel 用 `exp2` 配合 `sm_scale * LOG2E` 预缩放;BLOCK_K 按设备 shared memory 上限自适应收缩(gfx942 的 64 KiB 限制下 BF16 tile 从 64 降半)。调度层把 `triton` 加入 `--dsa-prefill-backend`/`--dsa-decode-backend` 的 `DSA_CHOICES`,启动时校验必须搭配 `--kv-cache-dtype fp8_e4m3`;HIP 上强制使用原始 MLA KV layout(512 nope + 64 rope)而非其他非 ROCm 后端的 scaled fp8 layout,并把 gfx950 fused-rope skip 路径扩展到 Triton。另按架构关闭 gfx950 上有害的 direct-to-LDS async copy 编译选项。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py: 重写 prefill,新增 split-dim 单遍 kernel 与 split-K/reduce kernel
python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py: 新增 decode 实现(见 PR 全文)
python/sglang/srt/layers/attention/dsa_backend.py: Triton prefill/decode dispatch,移除旧 env-var 门控 prefill 路径
python/sglang/srt/server_args.py / arg_groups/overrides.py: 暴露 triton 选项并校验 fp8 KV cache 约束
python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py: Triton DSA 选用 HIP 原始 MLA KV layout
```

```python
# 512-wide nope 按 4x128 分组,prefill 累加器:
qk = tl.dot(q0, tl.trans(kv0))
if NUM_GROUPS >= 2: qk += tl.dot(q1, tl.trans(kv1))
...
qk += tl.dot(q_tail, tl.trans(kv_tail))
```

## 性能声明

- 精度:GSM8K 400 题,TileLang 与 Triton 均 **95.5%**、0% invalid;精度同时在 gfx942 上验证。
- 隔离 prefill microbench:MI355X / GLM-5.2 shape(`H=16, D_V=512, D_TAIL=64, topk=2048`),split-dim prefill kernel 相比旧 512-wide-dot env 门控 Triton kernel **约 10% speedup**,cosine similarity 1.00000。
- 端到端:GLM-5.1-MXFP4 serving,TP4 MI355X,`ISL=8192, OSL=1024`,concurrency 2:TileLang 1005 total tok/s / 112 output tok/s / TTFT 834 ms / TPOT 17.1 ms → Triton **1044** / **116** / **701** / **16.6**(+3.9% 总吞吐、+3.6% 输出吞吐、-16.0% TTFT、-2.9% TPOT);正文称在测得的并发区间 Triton 比 TileLang 提升约 3–5%,且在所有并发、TP、长短上下文均有提升。

## 验证与回归风险

验证依赖 GSM8K 评估、prefill microbench(cosine similarity 1.00000)与端到端 bench。风险:仅支持 fp8 DSA 路径(启动校验强制 `--kv-cache-dtype fp8_e4m3`);prefill 要求 d_v 整除 128 且 H 为 2 的幂(不足 16 时 pad 到 16),不满足时显式报错并建议 `tilelang` 后端;kernel 消费 HIP 原始 fp8 KV layout,明确不支持 NVIDIA scaled fp8 layout;CI 状态 Base/Extra/AMD ROCm 10 三个 run 在 pack 记录中均显示失败(:x:),合并时是否已修复未注明。

## 关联

同区域:DSA/DeepSeek sparse attention 系列后端(TileLang、AITER);pack 未列出具体前后序 PR 编号,关联暂记"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
