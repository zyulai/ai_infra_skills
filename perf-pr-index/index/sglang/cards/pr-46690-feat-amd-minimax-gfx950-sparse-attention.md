# PR #46690: [Feat] [AMD] minimax: gfx950 sparse attention, NVFP4 KV and DSpark verify kernels

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kevin-mii |
| 合并 | 2026-11-01 |
| 分类 | Quantization (tier: major) |
| Label | quant, hicache, blackwell, run-ci, jit-kernel, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(22 个文件未收入),收录部分含 nvfp4 cache 定义、QK-norm+RoPE 融合 kernel、page_row.cuh |
| 链接 | https://github.com/sgl-project/sglang/pull/46690 |

## 一句话结论

为 MiniMax-M3.1-Flash 的 AMD gfx950 路径新增 Gluon/Triton 稀疏注意力 kernel 族:NVFP4 KV 稀疏 decode 降至 FP8 的 0.42–0.68x 时间、每 GB KV 承载 1.78x token,窗口化 draft verify 快 13–26%。

## 动机

MiniMax-M3.1-Flash 在 AMD MI350X(gfx950)上需要一整套针对其 sparse attention + DSpark draft 结构的 kernel:NVFP4 压缩 KV(MSA-v1 layout)的读取/写入、窗口化(非因果 draft block,window 4095)的投机 verify、fused prefill indexer、fp8 O epilogue 等。此前 gfx950 只有 Triton 通用路径,fp8 KV pool 的 prefill 也停留在 Triton kernel。本 PR 是 AMD 系列 kernel PR 之一(同系列 #46691 MoE/GEMM、#46692 quick all-reduce,集成在 #46693、#46694),沿用 DeepSeek-V4.1 AMD 的布局方式。

## 优化点

四类 kernel 机制:(1) NVFP4 KV(MSA-v1 layout):`Nvfp4Cache` 描述一行(token-head)= D/2 字节 e2m1 payload + D/16 个 E4M3 block scale,Trion 侧 `nvfp4_load_rows/even_odd` 用 gfx950 的 `v_cvt_scalef32_pk_f16_fp4` 单指令把一个 packed byte 转出 even/odd 两个 fp16 tile(e2m1 x E4M3 scale 至多 6 位有效位,fp16 精确),Gluon 侧按 u16 读取做同样 dequant;QK-norm+RoPE 融合 kernel 新增 tiled 版本,直接把 norm+RoPE 后的 K/V/index-K 量化 scatter 进 NVFP4 分页 cache。(2) 稀疏 decode:`decode/nvfp4_decode_gfx950.py` 在 Gluon 中对 packed NVFP4 K/V 做 sparse decode(one program per row,top-k chunk x KV head,per-warp online softmax)。(3) Verify:`verify_topk_idx.py` 给线性 draft chain 对共享 prefix 打分,`verify_topk_gfx950.py` 把 per-block max score 与 streaming top-k 融合在 MFMA tile 上;`verify_mla_window.py` 作为 `VerifyMLA` 子类,让 grouped-head split-KV verify 支持 DSpark 滑窗,一次 combine 完成,stage-1 window mask 为 constexpr(无窗口时编译消除)。(4) Prefill:`index_topk_gfx950.py` fused prefill indexer、`gluon_prefill_pair_gfx950.py` 每 program 处理两个 query 的 top-k block 并集、`nvfp4_prefill.py` 把 NVFP4 cache gather 成 Gluon prefill 的 paged layout;`common/out_quant_gfx950.py` fp8 O epilogue 让 decode/pair prefill 直接输出 fp8 o_proj 输入。HiCache 的 NVFP4 page-row copy 把 CTA quota 做成模板参数:HIP 上 32(MI350X 需 16+ 才能 saturate PCIe),CUDA 上维持 2。

## 关键改动

```text
python/sglang/kernels/ops/attention/minimax_sparse/common/nvfp4.py: 新增,Nvfp4Cache(MSA-v1 layout)与 Triton nvfp4 dequant 原语
python/sglang/kernels/ops/attention/minimax_m3_qk_norm_rope.py: 新增 gfx950 tiled fused QK-norm+RoPE+NVFP4/fp8 cache store kernel
python/sglang/kernels/ops/attention/minimax_sparse/decode/verify_mla_window.py: 新增,DSpark 滑窗 verify(一次 combine),主 verify_mla.py 仅加 constexpr window mask
python/sglang/kernels/ops/attention/minimax_sparse/prefill/: 新增 fused index top-k、two-queries-per-program Gluon prefill、nvfp4 paged gather
python/sglang/kernels/jit/csrc/kvcacheio/page_row.cuh: page-row copy 的 CTA quota 从常量 2 改为模板参数
```

```python
# NVFP4 block dequant(gfx950 单指令转换):
"v_cvt_scalef32_pk_f16_fp4 $0, $2, 1.0\n\tv_lshrrev_b32 $1, 16, $0"
```

## 性能声明

各 kernel 的 serving 增益在集成 PR 中测得(MI350X TP4,NVFP4 KV,DSpark acceptance 模拟为 3 tokens/step,32K-token contexts,逐项单独关闭,两组 GPU 取均值):

- Windowed draft verify(`verify_mla_window`):缺它时 decode TPOT **+20.6%**(1 request)、**+8.2–9.6%**(8–64);该窗口下 `num_stages=2` 比原实现 **快 13–26%**。
- Two-queries-per-program Gluon prefill:32K-token prefill **+10.3%**,decode 8–64 **+4.2–5.4%**。
- Gluon prefill 覆盖 fp8 KV pool(`SGLANG_MINIMAX_M3_AMD_KV_FORMAT=fp8`):prefill **+11.7%**(32K tokens)、**+12.4%**(8 x 8K requests)。
- NVFP4 KV 本身(kernel 级,CUDA-graph replay,per layer,110K context):verify indexer 从 8 requests 起耗时为 FP8 的 **0.75–0.84x**,sparse decode 为 **0.42–0.68x**,每 GB KV 承载 **1.78x** tokens。

CUDA 路径行为不变(topk_sparse 仅新增 Nvfp4Cache dispatch、page-row quota 保持 2);gfx950 上 M3 层切到新 kernel,QK-norm bf16 输出与 per-head kernel bitwise 一致,prefill 输出与 Triton kernel 差异在 bf16 rounding 内。

## 验证与回归风险

MI350X(ROCm 7.2 镜像,AITER acf8fdf93,SGLANG_USE_AITER=1)上 kernel 测试:`test_minimax_gfx950_prefill` 35 passed、`test_minimax_m3_qk_norm_rope` 55、`test_minimax_nvfp4_sparse_decode` 13、`test_minimax_verify_topk_idx` 68、`test_verify_shared_kv` 10;pre-commit clean。未在本分支单独执行的验证:CUDA(共享文件保持 main 行为)、端到端 serving(在集成 PR)。风险:gfx950 融合路径要求 head_dim=128/rotary_dim=64/neox,其他 ROCm GPU 走 per-head kernel(其主 K/V store 不再写 CUDA-graph padding slot 0,与 unfused store_cache 对齐);NVFP4 cache store 仅 gfx950 tiled kernel 支持;Qwen3.5/Kimi-K3 verify 输出声明 bitwise 不变,但 verify_mla.py 有共享改动。CI:Base 通过,Extra/AMD ROCm 10 失败(:x:)。

## 关联

系列前序/并行:#45936(被本系列取代,已关闭)、#46691(MoE/GEMM)、#46692(quick all-reduce)、#46693(M3.1 call sites 集成)、#46694(indexer CP);布局参照 DeepSeek-V4.1 AMD 系列 #41018–#41021 与 #41308。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
