# PR #27053: [BCG][GLM5] perf: BCG support and prefill enhancements

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mattteochen |
| 合并 | 2026-06-24 |
| 分类 | Other(tier: major) |
| Label | documentation, deepseek, run-ci, piecewise-cuda-graph, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未记录);diff 可用,另有 3 个文件被截断(详见 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/27053 |

## 一句话结论

给 NSA 模型接入 breakable CUDA graph 并重排 prefill 图切分,消除单 kernel graph island,1k1k B200 TP8 最高并发下较 PCG 快约 4%、TTFT 改善 10%。

## 动机

#23351 为 GLM5(`nsa_indexer`)引入 PCG(piecewise CUDA graph)后发现两个问题:一是当前实现跑完整 indexer 路径,而它并不能完全被 CUDA graph 捕获,实测把 fast indexer 放在 eager 模式跑反而更快;二是 PCG 的切分会产生只含 1 个 kernel(孤立 `bmm`)的 CUDA graph island,dispatch 开销纯属浪费。另外超过 2048 tokens 后 PCG 直接禁用,长 prefill 反正都是 eager 在跑。因此需要更合理的图切分边界与 BCG(breakable CUDA graph)支持。

## 优化点

四项改动:(1) 为 NSA 模型增加 BCG 支持,`dsa_backend.py` 等处的图内判断从 `is_in_tc_piecewise_cuda_graph` 扩展为 PCG/BCG 双检测;(2) 新增 `is_graph_dsa_split_op_surface`(非 speculative extend 且处于 PCG/BCG 内),在该 surface 上 DSA indexer 整体(q/k proj、head gate、k-cache store、topk)作为单个 eager split op 运行——`pcg_dsa_indexer_prefill_split` 注册为 split op,BCG 侧用 `eager_on_graph(True)` 包装为 `bcg_dsa_indexer_prefill_split`,输出契约是原地写入预分配的静态 padded `topk_result` 缓冲(-1 哨兵填充 padding 行),供下游被捕获的 graph 固定地址读取;(3) 在 `forward_mla.py` 注册 `bmm + unified_attention_with_output` 复合 split op(`MlaBmmFusionPlan` 预分配输出缓冲),避免只含孤立 bf16 `bmm` 的 1-kernel graph island,并把 `q_nope/q_pe/k_pe` 拆分上提使两个 split op 在 FX 中相邻;正文注明此项 opt in(env variable),diff 注释则说明该 surface 上两个 fusion 均默认开启(no feature flag);(4) 更新 breakable graph handler 合并相邻 eager Modules(PCG 版做过又删,因 PCG 计划弃用)。附带删除空 CUDA graph island 并隐藏 torch 的空图 warning。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa/dsa_indexer.py: 新增 pcg/bcg_dsa_indexer_prefill_split split op,graph 内 prefill 整体走 eager dispatcher
python/sglang/srt/layers/attention/dsa/utils.py: 新增 is_graph_dsa_split_op_surface 判定(PCG/BCG + 非spec extend)
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py: MlaBmmFusionPlan + bmm并入attention的复合split op
python/sglang/srt/layers/attention/dsa_backend.py: 图内强制 use_mha=False 的条件扩到 BCG
python/sglang/srt/environ.py: 新增 SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM 开关
```

split op 输出契约关键注释:

```python
#   - num_tokens: slice key/out_cache_loc to the unpadded count (the graph
#     runs at a static padded shape). None => full (eager) shape.
#   - topk_result: pre-allocated padded buffer to fill in place (a downstream
#     captured graph reads it at a fixed address).
```

## 性能声明

条件:B200 + INTEL XEON PLATINUM 8570(GLM5/DSA 模型)。1k1k B200 TP8:对 PCG 在 max concurrency 处 Pareto 约 4% speedup;median TTFT 在 max conc 改善 10%(不在 pareto 内)。表格数字:conc 256 时 out TPS/GPU 568.99 -> 592.42(1.04x),total TPS/GPU 1137.42 -> 1184.26(1.04x),TTFT 173.76 -> 157.74 ms(1.10x);conc 128 时 TPS/user 1.03x。8k1k:把 `--piecewise-cuda-graph-max-tokens` 提到 8192 后高吞吐区约 1% 提升(conc 256);默认 2k 到 8k 的显存开销为 2.71 GB/GPU。删除空 graph island 本身收益很小(batch 1k1k prefill latency 0.05460 -> 0.0540)。

## 验证与回归风险

精度:gpqa 6 次重复,PCG mean 0.863(mean latency 14037.766 s),BCG mean 0.854(mean latency 15453.834 s)。CI:Latest PR Test (Base) 与 (Extra) 均通过。风险:split op 的输出契约(topk_result 原地写、静态 padded 地址)依赖下游图按固定地址读取,LoRA 命中 `indexer.weights_proj` 时在图模式下直接 raise;`weights_proj_lora`、DSA prefill CP、fp8 w_kc、kv_b LoRA active、非 FORWARD_ABSORB_CORE backend 等场景都有显式 carve-out;BCG 在图内强制 MHA off 以保证正确性。

## 关联

前序:#23351(为 GLM5 引入 PCG)。相关:#11612(bmm_fp8 在 cublas 12.9 下的 fix,被注释引用)。正文注明 PCG 计划弃用,本 PR 的 eager-module 合并优化只落在 BCG 侧。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
