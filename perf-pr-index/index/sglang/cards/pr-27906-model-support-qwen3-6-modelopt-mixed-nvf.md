# PR #27906: [Model] Support Qwen3.6 ModelOpt mixed NVFP4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-07-06 |
| 分类 | Speculative Decoding(major,classify 给出;实际为量化 checkpoint 支持与 MTP 适配) |
| Label | quant, run-ci, run-ci-extra, release-highlight |
| Diff | pack 未提供 diffstat(diff 正文可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/27906 |

## 一句话结论

让 SGLang 能以 modelopt_mixed 路径加载官方 Qwen3.6-35B-A3B-NVFP4 checkpoint 并支持 MTP 推理。

## 动机

官方 `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint(closes #26967)与常规 `modelopt_fp4` 导出不同:`config.json` 报告 ModelOpt `MIXED_PRECISION`,按 `model.language_model.*` 等 wrapper 前缀逐层列出 FP8 / W4A16 NVFP4 条目,`lm_head` 也是 W4A16 NVFP4;FP8 KV cache 设置只存在于 `hf_quant_config.json`。此前这类 checkpoint 会被路由到 `w4afp8` 回退路径或因元数据缺失加载失败。作者同时说明:该官方 checkpoint 在验证中并未表现出相对常规 NVFP4 导出的精度或 serving 优势,本 PR 的价值主要在于支持官方发布格式。

## 优化点

四层适配:(1) 检测——`_parse_modelopt_quant_config` 不再只认 NemotronH 架构,而是检查 `MIXED_PRECISION` 配置中是否存在实际 `NVFP4` / `W4A16_NVFP4` 量化层条目,有则走 `modelopt_mixed`;(2) 元数据——`get_quant_config` 发现 `config.json` 的 quantization_config 缺 `quantized_layers` 或 KV cache 量化信息时,回退读文件版 `hf_quant_config.json`;(3) 层名解析——`_quantized_layer_prefix_candidates` 在 `model.language_model.*` / `language_model.model.*` / 顶层 `lm_head` 之间做候选前缀互查,把 checkpoint 层名映射到 SGLang 模块名;(4) 执行——新增 `ModelOptNvFp4A16LinearMethod`(packed NVFP4 权重 + fp16/bf16 激活,走 `apply_fp4_marlin_linear`),把 ModelOpt 量化的 `ParallelLMHead` 当量化线性层走量化 apply 路径算 logits,并接受标量 scale 张量(单元素参数 reshape)。MTP 侧:Qwen3.5 MTP body 在 `modelopt_fp4` 与 `modelopt_mixed` checkpoint 下保持不量化(测试 checkpoint 不量化 MTP 层),draft 模型通过 `set_lm_head_from_target` 直接共享 target 模型已量化的 lm_head 参数与运行时元数据,避免重复加载。

## 关键改动

```text
python/sglang/srt/layers/quantization/modelopt_quant.py: W4A16_NVFP4 识别、前缀候选解析、ModelOptNvFp4A16LinearMethod
python/sglang/srt/configs/model_config.py: MIXED_PRECISION + NVFP4 层条目判定为 modelopt_mixed
python/sglang/srt/model_loader/weight_utils.py: config.json 不完整时回退 hf_quant_config.json
python/sglang/srt/layers/logits_processor.py: should_apply_lm_head_quant_method 守卫
python/sglang/srt/models/qwen3_5_mtp.py + speculative/eagle_worker_v2.py: MTP 共享量化 lm_head
```

## 性能声明

测试条件:单节点 GB300(四个验证变体各占一卡),`bench_serving` random 数据集,input 1024 / output 1024,`max-concurrency 1`,8 prompts;MTP 用 EAGLE 3 步 / topk 1 / 4 draft tokens。对比官方 checkpoint 与常规 NVFP4 导出:常规导出 no MTP 快 1.97x(321.35 vs 163.50 output tok/s,E2E 低 49.1%,TPOT 低 50.3%,TTFT 高 5.1%);MTP 模式常规导出快 1.55x(512.99 vs 331.92 tok/s,TTFT 高 8.2%)。即官方 mixed checkpoint 的 serving 表现劣于常规 NVFP4 导出,PR 定位是格式兼容而非性能提升。GSM8K 四变体均为 0.971-0.973(官方 no MTP 0.973 / MTP 0.972;常规 no MTP 0.973 / MTP 0.971)。

## 验证与回归风险

`test/registered/unit/model_loader/test_modelopt_loader.py` 扩充:NemotronH NVFP4 层判定、Qwen W4A16_NVFP4 判定、`use_per_token_activation` 显式参数覆盖 env、lm_head 量化守卫的三种 runtime state(接受 Marlin 状态、拒绝 stale method、拒绝 dense head + stale attrs)。风险面:`w4afp8` 判定逻辑改动影响所有 `MIXED_PRECISION` checkpoint 的路由;`logits_processor` 的量化 lm_head apply 是全局路径,守卫错误会影响其他量化模型的 logits;`vocab_parallel_embedding.weight_loader` 对标量权重新增 reshape 分支;MTP 共享量化 lm_head 只在无 hot_token_id 时生效。

## 关联

Closes #26967;同区域:ModelOpt FP8/NVFP4 量化路径(同分类 quant label 的其他 PR);Qwen3.5 MTP 支持为本 PR 的前提。pack 未列出其他具体 PR 编号。
