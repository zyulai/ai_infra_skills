# PR #33723: [3/N] elastic-ep: Recapture decode CUDA graphs after scale-up

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zackyoray |
| 合并 | 2026-09-23 |
| 分类 | Speculative Decoding |
| Label | run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 12 个文件,另有 2 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/33723 |

## 一句话结论

让 Elastic EP 运行时 scale-up 后主 rank 与 joiner rank 重捕获 FULL decode CUDA graph,扩容后的集群不再退化为 eager 执行。

## 动机

Elastic EP 基础 scale-up(PR #30164)与 scale-up 后 EPLB(PR #30553)落地后,原有约束是 decode 与 prefill CUDA graph 必须完全禁用才能扩容——扩容后的新 rank 无法使用既有图,集群只能全程 eager,损失 CUDA graph 加速。本 PR 作为该系列第 3 步,在扩展拓扑提交前重建 FULL decode graph,使 scale-up 后保留图加速。

## 优化点

扩容流程重组:joiner rank 的初始 decode graph 捕获被推迟到进入扩展 world 之后(`capture_cuda_graphs` 增加 `finalize` / `defer_distributed_setup` 参数,eager warmup 也随 `run_warmup=False` 推迟,因为 warmup 会通信);扩容时序为——完成当前 sampling step(`_maybe_finalize_elastic_cuda_graph_scale` 挂在 forward_batch_generation 与 split-prefill 的采样之后)、释放旧图状态(`drop_elastic_cuda_graph_state`:backend.cleanup 前对每张图 `graph.reset()`、清全局 graph memory pool、gc、empty_cache)、以扩展后的 Mooncake WORLD 与 NIXL rank mask 重建 decode graph(`recapture_elastic_cuda_graph`,重置 `_kernel_warmed_up` 让 warmup 在 rendezvous 后所有 rank 重复)、全体同步后再提交 scale 操作。一致性保证:primary 与 joiner rank 必须解析出相同的 graph mode 与捕获配置——`ElasticCudaGraphConfig`(decode backend、capture_bs、compile_bs、attn_tp/cp size、attention backend、padding、TBO、capture_hidden_mode)经 SHA256 签名 all_gather 比对,不匹配则 fail scale;TCPStore 的 scale cohort 记录扩展为含 `cuda_graph_enabled` 的 msgspec 结构,两侧不一致同样失败。配套修复:FA3 MLA 增加 graph 稳定 metadata,DeepSeek MLA 路径要求 FA3(`validate_elastic_cuda_graph_recapture` 校验);`enable_num_token_non_padded` 对 elastic joiner 放开(join 后需要 graph padding mask);`can_use_dp_reduce_scatter` 校验固定 TP group 是否匹配当前 attention DP x TP 布局;NIXL rank mask 改用 `get_data_plane_ep_size()`(数据面在 syncing 阶段就使用 pending 尺寸)。

## 关键改动

```text
python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py: ElasticCudaGraphConfig/签名 all_gather、drop/recapture_elastic_cuda_graph、capture_cuda_graphs 的 defer/finalize 拆分
python/sglang/srt/model_executor/model_runner.py: joiner 推迟捕获、_finalize_scale_up 内 recapture、cohort 图配置一致性校验
python/sglang/srt/elastic_ep/elastic_ep.py: ScaleCohort(msgspec)携带 cuda_graph_enabled,get_data_plane_ep_size
python/sglang/srt/arg_groups/parallel_hook.py: elastic ep 门控放宽为 decode=full|disabled + 一组 recapture 不支持项断言
python/sglang/srt/runner_backend/full_cuda_graph_backend.py: cleanup 先 graph.reset() 再释放引用
```

## 性能声明

PR 未注明性能数字(定位为功能补全:scale-up 后恢复 CUDA graph 加速)。定性的代价信息:recapture 是同步的,配置更多 decode batch size 会加长 scale-finalization 暂停;手工测试用 `--mem-fraction-static 0.5` 以留出重捕获显存。验证:4 个手工 Elastic EP 测试(4to5、4to6、4to8、4to5to6)通过,每次捕获 decode bs 1/2/4/8/16 并检查扩容后生成、joiner 路由、input logprobs 与 GSM8K 正确性;重复 4-to-5-to-6 与直接 4-to-6 验证中所有参与 rank 均成功捕获 5 个 decode graph 尺寸。

## 验证与回归风险

新增单测 `test_elastic_cuda_graph_setup.py`(deferred capture 跳过 distributed setup、recapture 在 rendezvous 后重复 warmup、scale joiner 跳过 post-start budget refresh)与 full graph backend cleanup 的 reset 单测。已知限制(全部为 fail-loud 断言):仅支持 FULL decode graph recapture(prefill 与 piecewise 不支持,prefill graph 必须 disabled);两组必须同配置同有效捕获尺寸;MLA 需要 FA3;投机解码、embedding、diffusion、encoder-only、forward hooks、PDMux split-prefill 均不支持;部署需预留重捕获显存。风险面:graph pool 全局释放重建,若其他组件持有池内引用会失效;签名比对覆盖 9 个配置维度,新增影响捕获的配置需入签;CI Base/Extra 未全绿(bypass-fastfail)。

## 关联

前序:PR #30164(runtime EP scale-up)、PR #30553(scale-up 后 EPLB)、RFC issue #22788(Elastic EP Scale)。本 PR 为系列 [3/N]。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
