# PR #64352: [Model Loader] Direct reload: reset MoE expert padding at begin

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hao-aaron |
| 合并 | 2026-11-06 |
| 分类 | MoE(tier: major) |
| Label | ready, ci/build, gpt-oss, quantization, inkling |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/64352 |

## 一句话结论

Direct 权重重载(`VLLM_RELOAD_MODE=direct`)开始时清零 MoE expert padding,使热更新后的张量与冷启动逐位一致,消除 dummy 起步后 GSM8K 掉约 2 分的问题。

## 动机

MoE expert 参数常大于写入它的 checkpoint tensor:DeepEP 把 hidden size 向上取整,MXFP4/TRT-LLM/AITER backend 把 intermediate size 取整,block-aligned FP8 TP 对 shard 做 padding;loader 只拷 checkpoint 部分。Direct reload(#64336,direct weight reload 工作 #54477 的一部分)不做全量重建,于是 padding 区保留了上一次 post-processing 的字节(在多次 sync 间漂移);`load_format=dummy` 冷启动(RL 常规做法)后 padding 是随机值,重载后污染推理输出。

## 优化点

机制:冷启动记录 checkpoint 布局时(`record_checkpoint_layout`),由 `moe_config` 判定是否存在 padding(hidden/intermediate 超过 unpadded 值,或 block-aligned TP sharding),把需要清零的参数记为 `(module, name)` 列表——判定必须在 post-load processing(可能改写 `moe_config`)之前做,所以也覆盖 dummy 等跳过 loader 的冷启动和绕过 loader 的 load plan(#64418)。`begin_direct_reload` 在 view-back 后对列表内参数整体 `zero_()`,loader 随后覆写所有 checkpoint 区域。填充值全为 0:padding-only scale block 只乘以同被清零的权重,MXFP4/NVFP4 scale 本就创建为 0,与冷启动一致。`padding_fill_params()` 返回该层参数中 `ndim > 1 且 numel > 2*shape[0]` 的部分,per-expert 标量(如可能缺席 checkpoint 的 per-tensor/static input scale)被保留。代价:fill 会抹掉 checkpoint 区域,更新必须重发这些参数的全部 checkpoint tensor(direct delivery contract 已如此约定,#54477 的完整性检查将强制)。附带修复:layerwise 路径的 `materialize_meta_tensor` 改为分配后清零,否则 padding 保留未初始化内存。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/routed_experts.py: 新增 padding_fill_params() 按 moe_config 判定需清零参数
vllm/model_executor/model_loader/reload/direct.py: record_checkpoint_layout 记录填充列表;begin_direct_reload 清零
vllm/model_executor/model_loader/reload/meta.py: materialize_meta_tensor 分配后 .zero_()
```

## 性能声明

无吞吐优化声明;数字为正确性与冷启动无回归验证(GB200,`vllm bench serve` random 1024 in / 256 out, concurrency 64;GSM8K 5-shot greedy 1319 题):

- gpt-oss-20b(TP2, MXFP4, direct reload + IPC weight transfer):main 上每次 sync 后 168 个 expert tensor 与冷启动不一致且逐次漂移,dummy 起步后 288 个不一致、GSM8K 掉约 2 分;PR 后两种场景均为 0 个不一致,GSM8K 0.9022 / 0.8969(main 对照 0.9030 / 0.8825)。
- DeepSeek-V4.1-Flash(TP4, `FLASHINFER_TRTLLM_MXFP4_MXFP8`,reload from disk):main dummy 起步 GSM8K 0.8787 → 0.8400(逐次下滑),PR 0.8976 → 0.8999;differing 张量 160→0。
- Layerwise reload 修复前 main 上 GSM8K 0.9037 → 0.0 → 0.0(直接崩溃),加 zero fill 后 0.8954 → 0.9007 → 0.9030。
- 冷启动 A/B:GSM8K 差异在 1 分内,吞吐差异双向 ≤2.4%,KV cache 容量逐位相同(冷启动只多记录一次 fill 列表)。

## 验证与回归风险

新增 `tests/kernels/moe/test_fp8_tp_loading.py::test_direct_reload_after_dummy_start_resets_padding[block|static]`(intermediate 256→384,先用 7 填满再 reload,验证结果与零初始化加载逐位相等、per-expert 标量与 `e_score_correction_bias` 不被动);`test_padded_moe_reload_releases_each_layer` 把 `torch.empty_strided` 注入 NaN 填充以检验 padding 清零(不加修复 16 例中 8 例失败);相关套件 153 passed。风险:`begin_direct_reload` 会清零 checkpoint 区域,依赖"每次更新重发全部 checkpoint tensor"的交付契约,部分更新场景不适用;`materialize_meta_tensor` 清零影响所有 layerwise reload 分配;Qwen3 两配置(BF16/FP8)在 main 上已由 `_zero_intermediate_padding`/block-aligned TP loader 处理,PR 显示无副作用。

## 关联

同区域:#54477(direct weight reload 总纲)、#64336(direct reload 模式)、#64418(load plan)、#65353(streamed gpt-oss MXFP4 TP split)、#64690(Inkling AMD padding,由本 fill 覆盖)、#66010(验证时叠加的修复)。替代了本 PR 早先"loader 记录 padding 区域"的设计。
