# PR #66652: [Perf][DSv4.1] CuTe DSL mHC boundary for decode batches up to 336 tokens

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-11-10 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | deepseek |
| Diff | diffstat 缺失(pack 未提供 +/− 统计;pack 的 diff 另有 1 file 因篇幅被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/66652 |

## 一句话结论

DeepSeek-V4.1 shifted mHC boundary 换 CuTe DSL 单 kernel,decode 48–336 tokens 每步省 1.5–5.8%。

## 动机

DeepSeek-V4.1 decode 每步约 80 次 Mega mHC launch 处理 shifted mHC boundary(residual update、下一 sublayer 的 pre-mix、Sinkhorn、collapse、RMSNorm);decode batch sizes 下 DeepGEMM `sm100_mega_mhc_impl` 每次调用 13–19 µs,远高于其访存量应有的水平。

## 优化点

新增 CuTe DSL kernel,移植自 TensorRT-LLM 的 DeepSeek-V4.1 mHC boundary(NVIDIA/TensorRT-LLM#21438,Apache-2.0),仅保留 vLLM 用到的部分(其多 kernel 大 batch 路径、MXFP8 epilogue、deferred-Sinkhorn、NVFP4 kernels 未搬):16-CTA cluster 各拥有一个 token tile 并切分 hidden 维,CTA 经 distributed shared memory 交换 partial mixes 与 sums of squares,整个 boundary 一次 kernel 完成;mixer GEMM 的 A fragments 以 coalesced layout 从 L2 流式读取并提前 4 个 k-step prefetch,替代常驻 120 KB weight slice,每 CTA 只占半个 SM,14 clusters 同时驻留,一波覆盖 336 tokens(tile = ceil(M/14),24 个 tile size)。每个 `DeepseekV4DecoderLayer`(target 与 DSpark draft)注册非持久 buffer(`hc_attn_boundary_weight` / `hc_ffn_boundary_weight`),在 `process_weights_after_loading` 就地填充,使捕获的 graph 读到更新后的权重;`mhc_shifted_post_pre` 对 ≤336 tokens 取新 kernel,更大 batch 保持 Mega mHC(两波时该设计反而输)。精度:mixer weight 取整到 TF32 并拆成和精确的 BF16 高/低部,每个乘积等于 Mega mHC 的 TF32 乘积;归约定序、结果确定,仅累加顺序不同。

## 关键改动

```text
vllm/models/deepseek_v41/nvidia/ops/mhc_boundary.py(新增): 16-CTA cluster 单 kernel boundary;prepare_boundary_weight 权重重排
vllm/models/deepseek_v41/nvidia/ops/mhc.py: mhc_boundary_weight 工厂与 mhc_shifted_post_pre 分派(boundary_weight 存在时优先)
vllm/models/deepseek_v41/nvidia/model.py: 注册/填充 hc_attn/ffn_boundary_weight 并传入 forward
tests/kernels/mhc/test_mhc_kernels.py: 新增 test_mhc_boundary_correctness
```

## 性能声明

- `vllm bench forward`,DeepSeek-V4.1-Flash(dummy weights),DEP2 单 GB300 节点,`FLASHMLA_MEGA_ATTN_DSV41`、`deep_gemm_mega_moe`、`sparse_indexer_topk_backend=deep_select`、DSpark k=5(synthetic acceptance 3.51),KV length 65536:step time 48 tokens/rank 9047.1→8519.5 µs(−5.8%)、96 tokens 9537.3→9049.1(−5.1%)、336 tokens 13822.7→13612.0(−1.5%),即 48–336 tokens/rank 步时降 1.5–5.8%。
- Op 微基准(单 GB300,CUPTI,cold L2,hidden 5120,20 Sinkhorn iterations,每 token 数 10 rounds 取中位):1 token 13.4→6.85 µs;196 tokens 16.0→11.36;336 tokens 18.62→15.07;1–196 tokens 快 1.41–2.04x,197–336 tokens 快 1.13–1.47x,无任何一轮慢于 Mega mHC;>336 tokens 跑 Mega mHC(1.00–1.02x)。
- Kernel profile:每步 79 个 `sm100_mega_mhc_impl` launch(含 DSpark draft 层的 5 个)全部替换;96 tokens/rank 时 1149→706 µs。

## 验证与回归风险

`tests/kernels/mhc/test_mhc_kernels.py` 新增 `test_mhc_boundary_correctness`(num_tokens 1/17/98/128/196/197/252/336,需 SM100 + CuTe DSL,否则 skip),对照 TileLang 参考。updated residual 与 Mega mHC 在 1–336 每 token 数 bitwise identical;graph vs eager bitwise identical;GSM8K unchanged。风险:仅 SM100 family 且 `has_cutedsl()` 时启用(buffer 为 None 即走原路径);`MhcBoundaryKernel.supports` 要求 `hc_mult == 4`、`hidden % 512 == 0` 等;每个 tile 需在 capture 前的 eager run 中编译。

## 关联

NVIDIA/TensorRT-LLM#21438(移植来源,Apache-2.0)。pack 未列 vLLM 侧其他同区域 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
