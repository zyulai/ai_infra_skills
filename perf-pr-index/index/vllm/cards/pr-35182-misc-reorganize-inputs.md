# PR #35182: [Misc] Reorganize inputs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @DarkLight1337 |
| 合并 | 2026-03-25 |
| 分类 | Other (tier: major) |
| Label | documentation, performance, rocm, frontend, ready, v1, multi-modality, llama, qwen, deepseek, cpu |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 "129 file(s) dropped from diff" |
| 链接 | https://github.com/vllm-project/vllm/pull/35182 |

## 一句话结论

重整 inputs 模块命名与分层，Test Plan 与 Test Result 为空、正文未给出定量性能数据。

## 动机

正文 Purpose 说明：`ProcessorInputs -> EngineInput` 重命名以避免对其应输入位置的混淆；标准化引擎输入命名（`TokenInputs -> TokensInput`、`token_inputs -> tokens_input`、`EmbedsInputs -> EmbedsInput`、`SingletonInputs -> SingletonInput`）；拆分 `vllm.inputs.data`（LLM API inputs 进 `vllm.inputs.llm`、Engine inputs 进 `vllm.inputs.engine`、`split_enc_dec_inputs` 移到 `vllm.inputs.engine.split_enc_dec_input`）；把一批 MM input 定义移出 `vllm.multimodal.inputs` 以避免循环导入并更便于用户访问。属代码组织重构，正文未描述性能瓶颈或故障现象。

## 优化点

具体迁移包括：`ModalityData`、`MultiModalDataBuiltins`、`MultiModalDataDict`、`MultiModalUUIDDict` 移到 `vllm.inputs.llm`；`MultiModalHashes`、`MultiModalPlaceholderDict -> MultiModalPlaceholders`、`MultiModalInputs -> MultiModalInput`、`mm_inputs -> mm_input`、`MultiModalEncDecInputs -> MultiModalEncDecInput`、`mm_enc_dec_inputs -> mm_enc_dec_input` 移到 `vllm.inputs.engine`；同时更新相关变量名与文档。正文 Purpose 列表即全部改动说明，未涉及算法、kernel 或调度机制；Test Plan 与 Test Result 两节在正文中均为空标题，checklist 各项未勾选。

## 关键改动

```text
vllm/beam_search.py: ProcessorInputs/TokenInputs/mm_inputs 等改为 TokensInput/MultiModalInput/mm_input/EncoderDecoderInput/DecoderOnlyEngineInput
vllm/entrypoints/llm.py、vllm/entrypoints/chat_utils.py、vllm/engine/protocol.py: LLM/engine 入口改用新的 inputs 命名与 import 路径
vllm/entrypoints/openai/ 下 chat_completion/completion/engine/realtime/responses/speech_to_text 各 serving: 更新 MM input 类型引用
vllm/benchmarks/datasets.py 与 vllm/entrypoints/anthropic/serving.py: 更新 inputs 引用
tests/models/multimodal/generation/test_pixtral.py 与 tests/models/multimodal/processing/test_common.py: import 由 vllm.multimodal 改到 vllm.inputs、类型改名
```

关键代码（来自 `vllm/beam_search.py` diff）：

```diff
-    orig_prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs
+    orig_prompt: TokensInput | MultiModalInput | EncoderDecoderInput
```

## 性能声明

PR 正文未给出定量性能数据。正文 Test Plan 与 Test Result 均为空，未提供任何 benchmark 数字、测试命令或结果；PR 未注明测试条件（模型/硬件/精度/workload 均未给出）。

## 验证与回归风险

PR 正文未注明验证方式：Test Plan、Test Result 两节为空，checklist 各项未勾选。pack 的 diff 显示改动波及大量入口文件（`vllm/entrypoints/openai/` 下多个 serving、`beam_search.py`、`benchmarks/datasets.py` 等）与两个多模态测试文件，且 pack 末尾注明 129 file(s) dropped from diff，说明改动面很广；正文未说明可能受影响的模型、硬件或精度路径，也未给出回归风险论述。

## 关联

无。正文未给出任何前序/后续 PR 或同区域 PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
