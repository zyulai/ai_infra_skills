# PR #58046: [Mypy] Fix mypy typing for Qwen and Qianfan models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ashraf-bhuiyan |
| 合并 | 2026-09-26 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | speculative-decoding, ready, qwen, dflash |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(8 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/58046 |

## 一句话结论

为 Qwen/Qianfan 全系 24 个模型文件补齐 mypy 类型并移出 mypy 排除名单,属于类型/CI 基建 PR,不改变运行时行为与性能。

## 动机

issue #26533 记录了 vLLM 模型实现长期被排除在 mypy 检查之外的问题:`tools/pre_commit/mypy.py` 中存在 `r"vllm/model_executor/models/[qQ]"` 排除项,Qwen/Qianfan 家族的类型错误不被 CI 发现,重构与接口演进缺少静态保障。在原基线 `7c877062ac` 上,选中的 Q-model 文件共有 297 个 mypy 错误(历史数据,非当前 main 的计数)。

## 优化点

纯类型修复,无算法/内核/调度改动:覆盖 optional 配置与 multimodal 字段、tensor/container 与 cache 类型(如 `_rope_by_thw_cache` 的 `LRUCache` 泛型参数)、generic processing-info 类型(含 ASR 继承 `Qwen3OmniMoeThinkerMultiModalProcessor[Qwen3ASRProcessingInfo]`)、MoE mixin 属性声明(`language_model`、`num_local_physical_experts`、`audio_tower`/`visual` 等)、realtime async-generator protocol(`buffer_realtime_audio` 从 async 改为同步签名)。附带少量等价重写:三处 `**{...} if cfg else {}` 展开改为先构造 `attention_kwargs` 字典;`_parse_and_validate_image_input` 的 `if image_embeds is not None:` 改为 `else:` 等。仅保留 5 处文档化、行级的 `# type: ignore[override]`(4 处不兼容的 speculative-drafter/MTP forward 签名,1 处实例级覆盖类级 LoRA 映射 `is_3d_moe_weight: ClassVar[bool]`),未新增任何文件级 ignore 或替代排除项。

## 关键改动

```text
tools/pre_commit/mypy.py: 移除 r"vllm/model_executor/models/[qQ]" 排除项
vllm/model_executor/models/qwen2_5_vl.py / qwen2_vl.py / qwen2_5_omni_thinker.py: 多模态处理与 encoder metadata 类型修复
vllm/model_executor/models/qwen3_dflash.py / qwen3_dspark.py / qwen3_5_mtp.py: speculative drafter 类型与 assert 收窄
vllm/model_executor/models/interfaces.py: SupportsRealtime.buffer_realtime_audio 签名修正
```

## 性能声明

无性能声明,也不适用:类型标注与 assert 收窄在 Python 运行时无可测影响(新增 assert 仅在 `python -O` 之外的路径增加 O(1) 检查)。PR 未做 GPU/模型质量评测,原因是 NVIDIA driver 不可用;明确声明结果仅覆盖 CPU unit/processor 与 registry-import 层面。

## 验证与回归风险

mypy 3.10/3.11/3.12/3.13 对全部 24 个 Q-model 文件 + `interfaces.py` 通过;pre-commit(所有 applicable hooks)通过。运行时回归:151 passed(test_qwen3_vl_mrope、test_qwen3_omni、test_qwen3_asr_mrope、test_dflash_causality 等)+ 13 passed(omni/vl processing)+ 41 passed(registry imports,Qwen/Qianfan),合计 205 个 distinct tests。风险:`multimodal_config = model_config.multimodal_config` 改为 `get_multimodal_config()`、`buffer_realtime_audio` 签名变化等属于公开接口层面的收紧;Qwen2/Qwen3-ASR forced aligner 中 `lm_head = None` 的赋值路径经 `nn.Module` 类型重声明绕过,行为不变但屏蔽了后续检查。

## 关联

issue #26533(mypy 覆盖总任务);Kimi 家族对应工作 #45296(不同模型族,互不重叠);Transformers backend 类型工作 #54320。同 batch 相关:dflash/dspark 模型文件与 pr-54163、pr-51725 的 spec decode 路径。
