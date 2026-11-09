# PR #17784: Upgrade transformers==5.3.0

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-03-18 |
| 分类 | Other (tier: major;依赖升级与全仓适配) |
| Label | documentation, high priority, quant, dependencies, Multi-modal, deepseek, npu, run-ci, diffusion |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,但 66 个文件因超长未收录,仅见约 20 个文件片段) |
| 链接 | https://github.com/sgl-project/sglang/pull/17784 |

## 一句话结论

把 transformers 升级到 5.3.0 并完成全仓 RoPE/config/tokenizer/多模态适配;无性能声明,属依赖基线变更。

## 动机

响应 issue #17779,将 `transformers` 升级到 5.3.0(v5 API)。v5 改变了 rope 配置结构(`rope_scaling` -> `rope_parameters`,Gemma3 按层类型嵌套)、视觉编码器返回类型(`BaseModelOutputWithPooling`)、tokenizer 行为(`batch_decode` 拼接、`clean_up_tokenization` 移除)等,SGLang 需要整体适配才能继续跟进上游。本卡收录它是因其影响所有模型的执行基线,而非优化本身。

## 优化点

非性能优化,是兼容性重构:(1) 依赖:`transformers>=5.2.0`、`huggingface_hub>=1.0.0`,移除 `hf_transfer` 自动启用逻辑;(2) RoPE:数十个模型改读 `config.rope_parameters["rope_theta"]`,`get_rope` 各 scaling 类型改用 `_get_rope_param`(缺失键告警并给默认值),MLA 模型(DeepSeek/Bailing/Sarvam)统一为 `compute_mla_mscale_scaling`,Gemma3 处理 v5 按层类型嵌套的 rope_parameters 并自带 default-rope 计算;(3) 多模态:LLaVA 适配 `CLIPImageProcessorFast` 返回 Tensor,Qwen2.5-VL 编码器取 `pooler_output`,Qwen2 移除 `padding_idx`,Gemma3 v5 API 适配;(4) tokenizer/config:auto 模式显式 `use_fast=True`、`batch_decode` 逐 token 包列表(v5 会拼接)、InternVL 移除 `clean_up_tokenization` 调用、`is_deepseek_nsa` 兼容 dict config、GGUF 版本解析 workaround、embedding `input_ids=None` 崩溃修复、modelopt loader 的 AutoConfig 回退。

## 关键改动

```text
python/sglang/srt/layers/rotary_embedding/factory.py: _get_rope_param 容错读取 rope_scaling 键。
python/sglang/srt/configs/model_config.py: compute_mla_mscale_scaling、is_deepseek_nsa dict 兼容。
python/sglang/srt/managers/tokenizer_manager.py: batch_decode 逐 token 包列表修复 logprob 文本。
python/sglang/srt/models/*.py(数十个): rope_theta/rope_scaling 改读 rope_parameters。
python/sglang/srt/model_loader/weight_utils.py、check_env.py: 移除 hf_transfer。
python/sglang/multimodal_gen/...: llama/qwen3 编码器 rope、qwen2_5vl pooler_output、image processor 参数过滤。
```

tokenizer 行为修复示例:

```python
# v5 中 batch_decode([1,2,3]) 会把所有 token 拼成一个字符串
token_texts = self.tokenizer.batch_decode(
    [[idx] for idx in token_logprobs_idx]
)
```

## 性能声明

PR 正文未提供任何性能基准数据(perf claim 检测为 none);该 PR 的价值在依赖基线与兼容性。移除 `hf_transfer` 意味着大模型下载加速路径改变(huggingface_hub>=1.0.0 内建传输),PR 未给出下载耗时对比。

## 验证与回归风险

验证:PR 主体是修测试(测试导入路径、Qwen2.5-VL `.visual` -> `.model.visual`、InternVL meta-tensor `.item()` 与 `all_tied_weights_keys`、matryoshka embedding 对齐),diff 含多个测试文件改动,但正文未汇总通过率;无单测新增,checklist 的格式项勾选。回归风险显著且 PR 自报未解决项:(1) fp8 量化与 diffusers 不兼容(上游,未修);(2) MiniCPM-V-4 视觉嵌入损坏,模型看到 "text/slice" 而非图像(未修);(3) InternVL2.5-8B piecewise cuda graph 下 MGSM 精度掉到约 0.36(v5 特有,单 prompt 正确、并发评估下失败,未修);(4) DeepSeek-OCR 的 `is_deepseek_nsa` dict 崩溃(TODO 未勾,虽然 diff 显示已做 dict 兼容);(5) rope 参数缺失键改为告警+默认值,极端配置下可能静默改变 attention scaling,正文多处提示 "Check model accuracy"。

## 关联

依赖升级常伴随性能适配,同区域后续:PR #45399(GLM-5.3 Python 模板,transformers 5.19.0 环境验证)、PR #29708 等 multimodal 系列均建立在 v5 基线上;issue #17779 为动因。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
