# PR #25455: [NPU] MiMo-V2-Flash Adaptation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @iridiumine |
| 合并 | 2026-06-10 |
| 分类 | Model-specific Fast Path(major) |
| Label | quant, npu, run-ci |
| Diff | pack 未提供 diffstat(diff 正文可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/25455 |

## 一句话结论

在 Ascend NPU 上为 MiMo-V2-Flash 打通 MTP(EAGLE)推理,并发 64 下总吞吐 +39.58%。

## 动机

MiMo-V2-Flash 需要在 Ascend NPU(要求 CANN 9.0)上支持 Multi-Token Prediction。此前 NPU 后端的 KV pool、graph capture、MTP attention forward 都不满足该模型 MTP 工作流的要求:KV cache 写入需支持 FIA 模式与 PagedAttention,NPU Graph 需要能捕获/回放 multi-layer EAGLE 的 draft extend,attention kernel 需要匹配 MTP 推理逻辑与 NPU 融合注意力算子,graph 元数据初始化/回放需支持 MTP 的 block table 与序列长度管理。

## 优化点

四块适配:(1) `NPUMHATokenToKVPool` 把 k/v 拆成独立 buffer(支持 `v_head_dim != head_dim` 的层),FIA 模式下改为逐层 Python list 视图,避免 torch.compile 在 graph capture 时捕获整个多层 tensor 导致 OOM,同时 `set_kv_buffer` 按 `v_head_dim` 写入;(2) 新增 `MultiLayerEagleDraftExtendNpuGraphRunner`(继承 CUDA 版 runner,替换为 `torch.npu.NPUGraph`),回放时用单独线程并发更新 `actual_seq_kvlen` 再 `replay()`,multi-layer EAGLE worker 在 NPU 上自动选用该 runner;(3) `forward_mtp` 切到 `npu_fused_infer_attention_score_v2`(参数改名 `num_query_heads`/`actual_seq_qlen` 等),支持 SWA 层(`sparse_mode=4` + SWA block table)、`draft_extend_v2` 模式、`learnable_sink`,sinks 层在 graph decode 下走 FIA TND 路径;qk/v head dim 不等且非 SWA 的 decode graph 用 workspace + `.out` 变体避免 padding;(4) graph 元数据:arch map 增加 `TARGET_VERIFY` 属性(`actual_seq_kvlen`),MiMoV2 架构自动选 v2 更新属性;附带 SWA allocator 的 int64 索引修复与 `ascend_fuseep` a2a backend 识别。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py: forward_mtp/decode 切 v2 融合算子、SWA 与 sinks 支持
python/sglang/srt/hardware_backend/npu/memory_pool_npu.py: KV pool k/v 分离 + v_head_dim + 逐层视图
python/sglang/srt/hardware_backend/npu/graph_runner/multi_layer_eagle_draft_extend_npu_graph_runner.py: 新增 NPU graph 捕获/回放
python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py: NPU 上选用 NPU graph runner
python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py: TARGET_VERIFY 元数据属性
```

## 性能声明

测试条件:Ascend NPU,TP16,`--speculative-algorithm EAGLE`(3 步/topk 1/4 draft tokens,`--enable-multi-layer-eagle`),SGLANG_ENABLE_SPEC_V2=1 + overlap plan stream,bench_serving random,`--max-concurrency 64`,input 3500 / output 1500,64 prompts。对比同为 graph 模式的 MTP 关闭/开启:NTP→MTP 后 Benchmark Duration 76.84 s → 55.05 s(-28.36%),总 token 吞吐 4,164.43 → 5,812.81 tok/s(+39.58%),输出吞吐 1,249.33 → 1,743.84 tok/s(+39.58%),Mean E2E 76,809.42 → 47,662.43 ms(-37.95%),Mean TPOT 45.87 → 28.36 ms(-38.17%),Max ITL 8,366.23 → 238.69 ms(消除严重延迟抖动)。精度:GSM8K 200 例 mean_acc 0.97(MiMo-V2-Flash-BF16)。注意这是 MTP on/off 的对比而非 PR 与 main 的对比,PR 未提供非 NPU 平台数字。

## 验证与回归风险

PR 未新增单元测试(checklist 中"Add unit tests"未勾选),验证以 GSM8K 精度与 bench_serving 为主。风险面集中在 NPU 后端:`NPUMHATokenToKVPool` 的 buffer 布局改动影响所有 NPU MHA 模型的 KV 分配与写入;`npu_fused_infer_attention_score_v2` 替换 v1 是行为等价性依赖算子本身;SWA 索引 int64 转换与 `sparse_mode=4` 分支只影响 SWA 层路径;CUDA/NPU 共用的 multi-layer EAGLE worker 仅按 `is_npu` 选择 runner,非 NPU 路径不变。

## 关联

同区域:NPU 后端其他适配 PR、multi-layer EAGLE spec v2 工作流(MultiLayerEagleMultiStepDraftExtendCudaGraphRunner 为其 CUDA 基类)。pack 未列出其他具体 PR 编号。
