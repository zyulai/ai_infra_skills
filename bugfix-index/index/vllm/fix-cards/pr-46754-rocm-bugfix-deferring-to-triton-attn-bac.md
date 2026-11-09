# PR #46754: [ROCM][Bugfix] Deferring to TRITON_ATTN backend for Prefix lm models instead of ROCM_ATTN and fixed minor bug in Gemma3 Multimodal code

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46754](https://github.com/vllm-project/vllm/pull/46754) |
| 作者 | GodRishUniverse |
| 合并日期 | 2026-10-13 |
| labels | bug, ready, rocm, v1 |
| 热度 | 👍 2 · 💬 32 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
ROCm 两个 attention 后端谎报支持 mm_prefix,prefix-LM 多模态模型(如 Gemma3)静默用错 attention mask;移除虚假 override 让选择器回退 Triton,并修 Gemma3 多模态的 `is_mm_prefix_lm` 标志。

## 问题现象
ROCm 上 prefix-LM 多模态模型(如 Gemma3)的 attention 行为不正确:image token 没有按模型要求彼此做双向(bidirectional)attention,prefix-LM 多模态模型静默地以错误的 attention mask 运行。影响面:ROCm 平台上 `RocmAttentionBackend` / `RocmAiterUnifiedAttentionBackend` 被选中且模型需要 mm_prefix 的场景。修复后 lm-eval 文本任务基本持平(无回归),MMMU-val 两档模型均 +0.0066,lmms-eval 图文任务全面提升(Gemma3-4B:docvqa anls 0.5938→0.7124(+0.1186)、chartqa 0.2850→0.3480、ai2d 0.6750→0.7280、infovqa +0.0584、vqav2 +0.0079;Gemma3-12B 类似,仅 vqav2 -0.0127)。硬件:AMD Radeon AI PRO R9700。

## 根因
正文披露:`RocmAttentionBackend` 与 `RocmAiterUnifiedAttentionBackend` 都把 `supports_mm_prefix()` override 成返回 `True`,但两个后端实际都没有实现 prefill 阶段双向 image-token attention 所需的 `mm_prefix_range_tensor` 传参。于是后端选择器以为 ROCm 后端支持 mm_prefix,直接选用它,导致 prefix-LM 多模态模型带着错误 mask 静默出错。另有一处相关缺陷:Gemma3 多模态路径没有设置 `config.text_config.is_mm_prefix_lm = True`(text-only 路径不需要该标志,多模态路径需要),模型侧的 prefix-LM 意图未传给后端选择逻辑。

## 修复方案
正文列出 4 项改动(注:pack 的 diff 素材仅含其中两个文件,其余文件改动未包含在 diff 中):
1. `rocm_attn.py` / `rocm_aiter_unified_attn.py`:移除错误的 `supports_mm_prefix() -> True` override,两后端继承基类默认 `False`,后端选择器对 prefix-LM 模型正确回退到 Triton unified attention。
2. `gemma3_mm.py`:多模态构造函数中显式 `config.text_config.is_mm_prefix_lm = True`(diff 可见,附注释引用 HF gemma3 博客「image tokens attend to each other」):
   ```python
   config.text_config.is_mm_prefix_lm = True
   ```
3. `rocm.py`:把 CUDA 平台路径的 `disable_chunked_mm_input` guard 移植到 ROCm——chunked prefill 会把多模态 item 切开、破坏双向 attention 模式,prefix-LM 多模态模型下 ROCm 自动禁用,对齐 CUDA 行为(diff 未含此文件)。
4. 回归测试 `test_rocm_backends_do_not_support_mm_prefix`(位于 `tests/v1/attention/test_rocm_attention_backends_selection.py`,正文写的是 `tests/kernels/attention/test_rocm_attention_selector.py`),断言两个 ROCm 后端 `supports_mm_prefix()` 为 False。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:ROCm 平台 attention 后端选择:需要 mm_prefix 的多模态 prefix-LM 模型从 ROCm 原生后端改走 Triton unified attention(性能与数值行为随之后端变化);不需要 mm_prefix 的模型选择不变;Gemma3 多模态现在会被标记为 mm_prefix LM,并在 ROCm 上自动禁用 chunked mm 输入(吞吐可能受影响);CUDA 平台不受影响。

## 关联
正文未列出(回归测试信息串中引用 [vllm#48172](https://github.com/vllm-project/vllm/issues/48172) 与 [ROCm/aiter#4168](https://github.com/ROCm/aiter/issues/4168),出自 diff 而非正文)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
