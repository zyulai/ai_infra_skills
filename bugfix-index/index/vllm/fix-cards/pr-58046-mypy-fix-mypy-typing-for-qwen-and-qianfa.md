# PR #58046: [Mypy] Fix mypy typing for Qwen and Qianfan models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58046](https://github.com/vllm-project/vllm/pull/58046) |
| 作者 | ashraf-bhuiyan |
| 合并日期 | 2026-09-26 |
| labels | speculative-decoding, ready, qwen, dflash |
| 热度 | 👍 0 · 💬 38 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
为 Qwen/Qianfan 模型家族补齐 mypy 类型并把 `[qQ]` 从 mypy 排除名单移除,24 个 Q 系模型文件全部纳入类型检查(基线 297 个错误)。

## 问题现象
Qwen/Qianfan 模型文件长期被 mypy 排除(`tools/pre_commit/mypy.py` 的 `r"vllm/model_executor/models/[qQ]"`),类型债无人检查。作者在 upstream main 上移除排除后跑 `pre-commit run mypy-3.12` 复现:在原始基线 `7c877062ac` 上,所选 Q 系模型文件共有 297 个 mypy 错误(作者注明这是历史结果,非当前 main 的错误数)。该 PR 处理 issue #26533 的一部分,纯类型修复,无运行时行为变更。

## 根因
正文披露:Qwen/Qianfan 模型家族的 typing 未随接口演进而维护,且被 mypy 全目录排除掩盖。具体缺口包括:可选的配置与多模态字段、张量/容器与缓存类型(LRUCache 值类型等)、泛型 processing-info 类型(含 ASR 继承)、MoE mixin 属性缺失声明、realtime 的 async-generator 协议不一致(`buffer_realtime_audio` 被声明为 async 而基类协议非 async)。

## 修复方案
- `tools/pre_commit/mypy.py`:删除 `r"vllm/model_executor/models/[qQ]"` 排除项,24 个 Q 系模型文件全部纳入检查。
- `vllm/model_executor/models/interfaces.py`:`SupportsRealtime.buffer_realtime_audio` 由 `async def` 改为 `def` 以匹配协议。
- qianfan_ocr/qwen2/qwen2_moe/qwen3/qwen2_5_vl/qwen2_vl/qwen3_5 等 23 个文件:为可选配置(如 `quant_config: QuantizationConfig | None`)、多模态输入 dict、缓存与容器加类型注解;`get_multimodal_config()` 替代可能为 None 的 `multimodal_config` 属性访问;新增局部断言(`assert isinstance(..., torch.Tensor)` 等)收窄类型;`attention_kwargs` 提取为具名 dict 变量;`window_index`/`cu_seqlens` 等改用带类型的具体变量名。
- 保留 5 处带注释的行级 typing 例外(`# type: ignore[override]`):4 处不兼容的 speculative-drafter/MTP `forward` 覆盖(qwen3_5_mtp、qwen3_dflash 等)与 1 处类级 LoRA 映射的实例局部覆盖;未新增 file-wide ignore 或替代排除。
- 注:pack 中 diff 在 40000 字符处截断(qwen3_dflash.py 之后的改动未见完整内容),以上仅覆盖截断前可见部分。

## 影响与回归风险
- 回归面:纯类型与局部变量重命名级改动,运行时行为应不变;`buffer_realtime_audio` 从 `async def` 改 `def` 属接口签名变化,若有子类按 async 覆写需对齐。验证:mypy 3.10-3.13 对 24 个文件 + interfaces.py 全部通过;CPU 上 205 个相关单测通过(test_qwen3_vl_mrope、test_qwen3_omni、dflash causality、registry imports 等);正文说明因无 NVIDIA 驱动未做 GPU 与模型质量验证。

## 关联
#26533(部分处理)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
