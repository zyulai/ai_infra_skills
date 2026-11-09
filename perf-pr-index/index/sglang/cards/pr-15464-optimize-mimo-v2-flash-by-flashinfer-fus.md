# PR #15464: Optimize MiMo-V2-Flash by flashinfer fused allreduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-12-20 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/15464 |

## 一句话结论
让 MiMo-V2-Flash 接入 FlashInfer fused_allreduce(allreduce+rmsnorm+residual_add 融合),E2E TTFT 降低 5.1%。

## 动机
MiMo-V2-Flash 在 TP 下每层 MLP/attention 输出走独立的 allreduce,再各自接 rmsnorm 与 residual add;`--enable-flashinfer-allreduce-fusion` 提供的 FlashInfer fused_allreduce 可把 allreduce+rmsnorm+residual_add 融合为一次操作,但该模型尚未接入。本 PR 的目标即让该模型利用这一融合路径。

## 优化点
`mimo_v2_flash.py` 接入已有的 layer communicator 融合协议:MLP `forward` 增加 `should_allreduce_fusion`/`use_reduce_scatter` 参数,`down_proj` 在融合或 reduce-scatter 时 `skip_all_reduce`;`forward_normal` 在融合/reduce-scatter/fp4-allgather 情况下跳过尾部 `tensor_model_parallel_all_reduce`。DecoderLayer.forward 计算 `should_fuse_mlp_allreduce_with_next_layer`,融合时给 hidden_states 打 `_sglang_needs_allreduce_fusion` 标记并跳过 `postprocess_layer`(由下一层融合完成 allreduce+rmsnorm+residual),否则走原 postprocess。同时为 `LayerCommunicator` 开启 `allow_reduce_scatter=True`(DP padding 场景用 reduce-scatter 替代 all-reduce)与 `is_last_layer` 判定,并新增 EAGLE3 层捕获(`set_eagle3_layers_to_capture` + `prepare_attn_and_capture_last_layer_outputs`)。

## 关键改动

```text
python/sglang/srt/models/mimo_v2_flash.py: MLP/DecoderLayer 接入 should_allreduce_fusion / use_reduce_scatter,跳过独立 allreduce
```

## 性能声明
E2E TTFT reduce 5.1%(原文数字,未附 A/B 明细表)。测试启动命令:`--model XiaomiMiMo/MiMo-V2-Flash --tp-size 4 --attention-backend triton --disable-radix-cache --trust-remote-code --enable-flashinfer-allreduce-fusion`(fp8 checkpoint、bfloat16 dtype 见启动日志);PR 未注明评测用的 workload 与硬件型号。

## 验证与回归风险
diff 中未包含新增测试;正文以服务器启动日志展示 fused allreduce workspace/lamport 初始化成功。风险:融合路径由 `should_fuse_mlp_allreduce_with_next_layer` 逐层判定,`_sglang_needs_allreduce_fusion` 标记把 allreduce+rmsnorm+residual 的责任移交给下一层,最后一层由 `is_last_layer` 排除;DP padding 下的 reduce-scatter 与 fp4 allgather 分支同时新增,行为组合需在对应拓扑下验证;EAGLE3 capture 改动与本优化同 diff 携带,验证面较大。

## 关联
依赖 `--enable-flashinfer-allreduce-fusion` 服务端能力;同区域改动(allreduce 抽象/capture)见 #3709(素材未标注直接关系)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
