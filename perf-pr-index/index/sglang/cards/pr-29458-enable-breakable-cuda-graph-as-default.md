# PR #29458: Enable Breakable Cuda Graph as Default

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Oasis-Git |
| 合并 | 2026-07-02 |
| 分类 | CUDA Graph / Compiler |
| Label | high priority, blackwell, run-ci, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 10 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/29458 |

## 一句话结论

把 prefill CUDA graph 的默认后端在 CUDA 平台上从 tc_piecewise 切换为 breakable(BCG),让默认配置即获得分段捕获的 prefill 图加速。

## 动机

PR 正文为空模板,动机需从 diff 推断:prefill CUDA graph 已有 breakable 与 tc_piecewise 两个后端,默认值一直是 tc_piecewise(torch.compile 路径,捕获慢、依赖编译);BCG 分段捕获不依赖 torch.compile,捕获更快且已被 Blackwell 验证(labels 含 blackwell、release-highlight),故将其设为 CUDA 平台默认。注意:此结论来自代码变更本身,PR 正文未陈述。

## 优化点

核心变更一处:`CudaGraphConfig.prefill` 的默认 backend 经新增的 `default_prefill_backend()` 决定——CUDA 上为 `BREAKABLE`,其他平台(HIP/NPU)保持 `TC_PIECEWISE` 直至 BCG 在这些平台验证通过;lazy import 保持该模块 stdlib-only 的导入约束。配套工作:失败提示重构为模板 `PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG(backend, suggestions)`,BCG 与 TCPCG 各有编号建议列表(切后端 / 关闭 / 调 mem-fraction),捕获异常在 context manager 中记录后 re-raise;`_disable_breakable_cudagraph_if_incompatible` 大幅扩充不兼容规则——MLA attention、DeepSeek-V4(c4 indexer scratch pinned 在 capture pool 导致 OOM)、context parallel(attn_cp_size>1 的 all_gather replay 尺寸不匹配)、LoRA(捕获 + adapter 权重超 host RAM)、MoE A2A 后端(bucket 尺寸超 FlashInfer dispatch cap)、DP attention(未验证)、多模态模型(replay fault);BCG 捕获时强制 `input_embeds=None`(BCG 图仅文本,多模态 embeds 需走 tc_piecewise 否则 NaN logits);BCG layer_model 解析扩展支持 `language_model.layers` 形态;移除 runner 构造里对 capture 的 try/except 包装(异常由 context manager 记录)。

## 关键改动

```text
python/sglang/srt/model_executor/cuda_graph_config.py: default_prefill_backend() — CUDA 默认 BREAKABLE,其他平台 TC_PIECEWISE
python/sglang/srt/server_args.py: BCG 不兼容规则扩充(MLA/DSV4/CP/LoRA/MoE A2A/DP-attn/多模态)
python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py: BCG input_embeds 强制 None;layer_model 解析扩展
python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py: BCG_FAILURE_HINT 与异常日志
```

## 性能声明

PR 正文未注明任何 benchmark 数字(Speed Tests / Accuracy Tests 章节均为空模板)。定性收益为默认 prefill CUDA graph 从 torch.compile 路径换到分段捕获路径;具体加速比 PR 未提供。

## 验证与回归风险

两个已注册测试做了 BCG 兼容性适配:`test_fp8_blockwise_gemm.py` 显式 pin `--cuda-graph-backend-prefill tc_piecewise`,TODO 注明"默认 breakable 下 MXFP8 RMSNorm 以 bf16 运行、损失精度,与本 PR 无关";`test_nvfp4_gemm.py` gsm8k 阈值从 0.64 降到 0.63(TODO:BCG-prefill RMSNorm fp32 修复落地后恢复)。风险面:默认切换把 BCG 的已知限制带给所有 CUDA 用户——MLA 模型(DSV3/R1 系)、DSV4、CP、LoRA、MoE A2A、DP attention、多模态全部自动降级为 tc_piecewise 或 disabled(有 warning);MXFP8/NVFP4 精度回归尚未修复(阈值被临时放宽);BCG 与 EAGLE 组合在 runner 构造期被直接拒绝。

## 关联

同区域:PR #27988(full prefill CUDA graph,复用本 PR 同期的 prefill runner)、PR #30206(BCG 捕获的多请求 bucket 修复)、PR #33723(elastic-ep 下 decode CUDA graph 重捕获)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
