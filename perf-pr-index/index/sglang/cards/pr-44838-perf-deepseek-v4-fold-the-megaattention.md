# PR #44838: [Perf] deepseek_v4: fold the MegaAttention query layout into wq_b for block-FP8-as-MXFP8 checkpoints

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kpham-sgl |
| 合并 | 2026-10-17 |
| 分类 | Quantization |
| Label | deepseek |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);单文件 deepseek_v4.py,约 +28/-6 行 |
| 链接 | https://github.com/sgl-project/sglang/pull/44838 |

## 一句话结论

把 DeepSeek-V4.1-Flash block-FP8 checkpoint 的 MegaAttention query 布局折叠进 wq_b 权重,省掉每层 prefill 时 33ms/步的 pack_query 填零与跨步拷贝,16K prefill chunk 247→235ms。

## 动机

`setup_mega_attention` 只在 `wq_b` 为原生 MXFP8 linear 时才把 MegaAttention 的 query head 布局折叠进 `wq_b` 行(`interleave_query_rows`)。DeepSeek-V4.1-Flash 发布的是 `[32, 32]` block-FP8 checkpoint,其 dense linear 经 block-FP8-as-MXFP8 视图(`block_fp8_mxfp8_ready`)走同一 FlashInfer MXFP8 GEMM,于是 `mega_wq_b_permuted` 保持 false,每层每次 forward 都要 `pack_query`:对 `[tokens, 64, 512]` bf16 query 做 `new_zeros` 再跨步转置拷贝。GB300 上 torch profiler 实测每个 16K step(rank 0):`aten::new_zeros` 40 次 × 141 µs = 5.7 ms,`aten::copy_`(strided direct_copy)40 次 × 691 µs = 27.6 ms——占 322 ms step 的 10%,且 80 次 launch 落在一个 dispatch-bound(中位 inter-kernel gap 41 µs)的 eager step 上;每层写约 1.2 GB 零、经非连续拷贝搬运约 2.4 GB。

## 优化点

`_wq_b_uses_flashinfer_mxfp8` 的判定从仅认 `use_mxfp8` 扩展为同时接受 `block_fp8_as_mxfp8`(FlashInfer CUTLASS / CuTe-DSL backend),使该 checkpoint 的 `mega_wq_b_permuted` 置真。`setup_mega_attention` 中按 quant method 分流:原生 MXFP8 走原路径不变;block-FP8 情况下,因 `interleave_query_rows` 以 16 行组为单位移动 query 行,把 `[32, 32]` 的 block scale 沿 N 维 `repeat_interleave(block_n // 16)` 精化到 16 行,再按与权重相同的置换重排,重新绑定为 `weight_scale_inv` 并把 `weight_block_size` 改为 `[16, block_k]`;权重本身做 `interleave_query_rows`。之后既有的 post-load pass 会像任何 block-FP8-as-MXFP8 layer 一样构建 swizzled MXFP8 scale,`pack_query` 被完全跳过。数值上:到达 GEMM 的 FP8 字节与 e8m0 scale 完全相同,只是行顺序变为 attention kernel 本就消费的顺序;32 行块 scale 精化到 16 行块是精确操作。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: _wq_b_uses_flashinfer_mxfp8 认可 block_fp8_as_mxfp8;setup_mega_attention 增加 block-FP8 分支(scale 精化 + 权重置换)
```

关键机制(block scale 精化到 16 行并随权重重排):

```python
block_n, block_k = method.weight_block_size
sf = self.wq_b.weight_scale_inv.data.repeat_interleave(block_n // 16, dim=0)
sf = (sf.view(self.n_local_heads, 32, -1).transpose(0, 1).reshape(sf.shape))
self.wq_b.weight.data.copy_(
    interleave_query_rows(self.wq_b.weight.data, self.n_local_heads))
self.wq_b.weight_scale_inv.data = sf.contiguous()
method.weight_block_size = [16, block_k]
```

## 性能声明

GB300,DEP2(TP2 + DP attention,EP2),DeepSeek-V4.1-Flash,MegaMoE,Mega-mHC,decoder bounded replay,eager prefill,每 rank 16K prefill budget,rank 0,main(job 3470)对比本 PR(job 3476):

- 单个 16K-token 冷 chunk(3 次取中位):247 → 235 ms;128K prompt 分 8 个 16K chunk:1906 → 1764 ms(约 7.4%)。
- Profiled 16K step:`new_zeros` + strided `copy_` 每步 80 次(33.9 ms)→ **0**;每步 kernel launch 数 1847 → 1756;attention section 每层中位(mHC kernel 到 MoE block 的 mHC kernel)5.03 → 3.39 ms(layer 6:5.00 → 3.36 ms,17 → 15 launches)。
- 注:同 run 的 cached-prefix 数据在多次运行间双向波动达 15%,作者未报告;GSM8K native-acceptance gate 当时未跑。

## 验证与回归风险

PR 未新增测试;CI 状态显示 PR Test Base、Extra、AMD ROCm 10 均失败(x),合并时点存在未绿 CI。风险点:对 `[32,32]` block scale 的 16 行精化与重排只精确适用于该 checkpoint 形状(n_local_heads ∈ (64,128) 的判定保留);`weight_block_size` 运行时改为 `[16, block_k]`,其他依赖该属性的路径需兼容;依赖 `mxfp8_dense_backend` 为 FlashInfer CUTLASS/CuTe-DSL,其他 quant backend 路径不变。

## 关联

同区域(DeepSeek-V4 / MegaAttention / MXFP8):#31470(flashinfer mega-moe)、#30805(trt-llm dsv4 attention)、#43736(dsv4.1 SWA decoder replay)、#46024(MHC mxfp8 norm)。同作者标注的启动日志:`MegaAttention wq_b query-layout permute: applied=True (block-fp8 served as mxfp8)`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
