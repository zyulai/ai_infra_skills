# PR #43865: [Fix] tokenizer_manager: Prevent scheduler crash and has_audio_understanding misreport for audio-declaring models without an audio tower

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#43865](https://github.com/sgl-project/sglang/pull/43865) |
| 作者 | enternal111 |
| 合并日期 | 2026-11-02 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
声明 audio token 但无 audio tower 的模型收到音频请求会杀死 scheduler 进程并误报音频能力;tokenizer_manager 提前 400 拒绝,`is_audio_understandable_model` 改按值判空。

## 问题现象
向「multimodal processor 定义了 audio token 但模型没有 audio tower」的模型(如 `gemma-4-31B-it`,其 `config.json` 带 `audio_config: null`)发送音频请求,不会在 API 层被拒绝;请求进入 scheduler 后 `get_audio_feature()` 在 `model.forward` 内抛异常,整个 scheduler 进程死亡。同时 `/get_model_info` 对 `gemma-4-31B-it` 误报 `has_audio_understanding: true`。崩溃(问题 1)仅影响 processor 定义 audio token 而模型无 audio tower 的模型;误报(问题 2)影响任何 `config.json` 带 `audio_config: null` 的模型。

## 根因
PR 正文披露两个底层问题:
1. 请求校验只检查 `is_multimodal`,不检查 per-modality 能力,音频请求通过了 tokenization/dispatch,直到 scheduler 深处才失败,杀死 scheduler 进程。
2. `ModelConfig.is_audio_understandable_model` 使用 `hasattr(hf_config, "audio_config")`;由于 config key 存在但值为 `null`,`hasattr` 返回 True,即使没有 audio tower,模型也被误报为音频模型。

## 修复方案
- `python/sglang/srt/configs/model_config.py`:`is_audio_understandable_model` 改为要求 `audio_config` 存在且非 `None`(顶层与 `thinker_config` 下都检查),替代 `hasattr`:
```diff
-                hasattr(self.hf_config, "audio_config")
-                or hasattr(
-                    getattr(self.hf_config, "thinker_config", None), "audio_config"
+                getattr(self.hf_config, "audio_config", None) is not None
+                or getattr(
+                    getattr(self.hf_config, "thinker_config", None),
+                    "audio_config",
+                    None,
                 )
+                is not None
```
- `python/sglang/srt/managers/tokenizer_manager.py` 的 `_tokenize_one_request`:当请求携带 audio data、processor 声明 audio token、且顶层 `audio_config` 显式为 None 时,提前 `raise ValueError(f"Audio inputs are not supported by {self.served_model_name}")`,以 400 拒绝,音频不再到达 scheduler。
- 新增单测 `test/registered/unit/configs/test_model_config.py::TestAudioUnderstandableFlag` 与 `test/registered/unit/managers/test_tokenizer_manager_audio_rejection.py`。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:capability flag 对 null audio_config 模型的报告由 true 变 false;tokenizer_manager 原生 `/generate` 路径存在一个有意的行为变化(正文 Limitations):过去「带合法 `audio_data` 且 prompt 无 audio placeholder」的请求会成功(base processor 丢弃多余音频并警告),现在被同样 400 拒绝;OpenAI 路径不受影响(总是插入 placeholder,这类请求以前会崩 scheduler)。processor 不定义 audio token 的模型(如 `glm-5.3-flash`)行为完全不变。其他 modality 的同形问题不在本 PR 范围。
- 正文验证:`gemma-4-31B-it`(4x H20)音频请求返回干净的单请求错误 `{"...","code":400}`,引擎继续服务;`glm-5.3-flash` 音频请求仍返回 200,与改动前一致。

## 关联
- [#43864](https://github.com/sgl-project/sglang/issues/43864)(本 PR 修复该 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
