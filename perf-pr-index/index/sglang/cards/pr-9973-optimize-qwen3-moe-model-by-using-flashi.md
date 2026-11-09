# PR #9973: Optimize Qwen3-moe model by using flashinfer fused allreduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-09-04 |
| 分类 | MoE(major) |
| Label | (none) |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/9973 |

## 一句话结论

让 Qwen3-MoE 复用 FlashInfer fused allreduce(allreduce+rmsnorm+residual_add),E2E 输入吞吐 +2.2%。

## 动机

TP 推理中,MLP 输出的 tensor_model_parallel_all_reduce 与下一层的 fused add RMSNorm 是两个独立访存密集操作。profiling 显示融合前 AllReduce 占 13.26%、FusedNormAdd 占 6.45%,合计 19.71%;用 FlashInfer fused_allreduce 融合后只剩 Fused AllReduce_Norm_Residual 12.98%。SGLang communicator 已有该融合基础设施,但 Qwen3-MoE 未接入,且门控仅限 SM100、token 数上限 2048。

## 优化点

三处改动把 Qwen3-MoE 接入融合路径。其一,`communicator.py` 的融合门控从仅 SM100 放宽到 SM90 或 SM100,token 数上限从 2048 提到 4096。其二,`qwen3_moe.py` 的 decoder layer forward 通过 `should_fuse_mlp_allreduce_with_next_layer(forward_batch)` 预判本层 MLP 的 allreduce 是否会与下一层融合:若会,则把 `should_allreduce_fusion` 传入 MLP,`forward_normal` 跳过 MLP 自己的 `tensor_model_parallel_all_reduce`(同时跳过条件还含 `should_use_flashinfer_cutlass_moe_fp4_allgather`),并在返回的 hidden_states 上打 `hidden_states._sglang_needs_allreduce_fusion = True` 标记、不再走 `postprocess_layer`,把 allreduce+rmsnorm+residual 推迟到下一层的 `forward_with_allreduce_fusion` 一并执行;构造 `LayerCommunicator` 时传入 `is_last_layer`(最后一层不能与"下一层"融合)。其三,`qwen2_moe.py` 的 `Qwen2MoeMLP.forward`(Qwen3-MoE 复用)增加 `should_allreduce_fusion` 参数并转给 `down_proj` 的 `skip_all_reduce`。

## 关键改动

```text
python/sglang/srt/layers/communicator.py: 融合门控放宽到 SM90/SM100,
  max token 2048 -> 4096
python/sglang/srt/models/qwen3_moe.py: should_allreduce_fusion 判定,
  跳过 MLP 自身 allreduce,标记 _sglang_needs_allreduce_fusion,
  is_last_layer 传入 communicator
python/sglang/srt/models/qwen2_moe.py: Qwen2MoeMLP 透传
  should_allreduce_fusion 至 down_proj skip_all_reduce
```

## 性能声明

测试条件:Qwen3-30B-A3B(小 MoE 模型),flashinfer attention,TP8,禁用 radix cache,PR 侧加 `--enable-flashinfer-allreduce-fusion`;`bench_serving`(sglang-oai backend)random 数据集,输入 4000/输出 1,flush cache 后各测 4 档并发。E2E 输入吞吐 +2.2%(作者注明当前用小 MoE 模型所以提升有限,更大的 235B 模型收益应更好):

- 并发 1:输入吞吐 51,354.78 vs 50,351.71 tok/s;mean TTFT 77.30 vs 78.85 ms。
- 并发 32:66,705.62 vs 66,295.27 tok/s;并发 64:66,678.73 vs 66,609.52 tok/s;并发 128:66,709.73 vs 66,046.91 tok/s。

profiling:融合前 AllReduce 13.26% + FusedNormAdd 6.45% = 19.71%;融合后 Fused AllReduce_Norm_Residual 12.98%(测试硬件未注明)。

## 验证与回归风险

精度:gsm8k(`benchmark/gsm8k/bench_sglang.py`,200 题,8-shot,parallel 128):PR 侧 0.730 vs main 0.695(注意两组的 data-path 均为 ./test.jsonl,非标准 gsm8k 测试集,数值不能直接与官方分数对比);另附一次 test_openai.py 输出样例。Checklist 全部未勾选,pack diff 未含新增单测。风险:融合标记 `_sglang_needs_allreduce_fusion` 依赖"MLP 输出直接进下一层 layernorm"的假设,最后一层由 `is_last_layer` 排除,但 DP/padding、reduce_scatter 与融合的互斥分支(`should_allreduce_fusion` 时不走 `postprocess_layer`)若与 communicator 状态不一致会漏做 allreduce;SM90 放开和 token 上限 4096 会影响所有已启用 `--enable-flashinfer-allreduce-fusion` 的模型路径。

## 关联

同区域(FlashInfer allreduce fusion 后续演化):#43639(MiMo 自动启用 CuTeDSL 后端)、#23402(MNNVL backend)、#47354 中提及的 GLM-5 部署使用 `--enable-flashinfer-allreduce-fusion`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
