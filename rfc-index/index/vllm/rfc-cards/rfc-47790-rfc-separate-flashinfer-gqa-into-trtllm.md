# RFC #47790: [RFC]: Separate Flashinfer GQA into TRTLLM and flashinfer backends.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47790](https://github.com/vllm-project/vllm/issues/47790) |
| 作者 | pavanimajety |
| 创建日期 | 2026-07-07 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 2 · 💬 9 |
| 分类 | Attention / Kernel |

## 一句话结论
提议把过于臃肿的 flashinfer attention 后端文件拆分为 flashinfer_gqa 与 trtllm_attention_gqa 两个后端,降低新增功能的复杂度。

## 提案动机
`vllm/v1/attention/backends/flashinfer.py` 已经膨胀、支持过多 kernel,各 kernel 特性差异大(标准 attention 支持矩阵见官方文档);且每个 wrapper 内部还包含多种 backend,导致新增任何新特性都过于复杂。正文详细列出原生 FlashInfer `BatchPrefill*Wrapper`/`*BatchDecode*Wrapper` API 与 TRTLLM API(如 `trtllm_batch_context_with_kv_cache`、`trtllm_batch_decode_with_kv_cache` 的 xqa/trtllm-gen 变体)在 FP8 compute、FP8 KV、SWA、NVFP4 KV、LSE 返回及 SM 架构上的支持差异。另指出:`backend="auto"` 只在 FA3 资格检查通过时选 FA3,否则 FA2,不会自动选 cuDNN/cutlass/cute-dsl/trtllm-gen;NVFP4 KV cache 支持 backend=fa2 或 trtllm-gen,当前代码只接了 trtllm-gen;fa2 路径下 NVFP4 KV 会被反量化为 bfloat16/float16 参与计算(flashinfer 0.6.13 起为 pre-Blackwell 兼容所设,目前 vLLM 未支持该路径)。

## 提出的方案
分阶段实施:
1. 就地拆分后端,实现干净分离并做第一阶段测试(关联 PR #47786);
2. 增加 flashinfer_gqa api + DCP + spec decode 支持;
3. 拆分为两个文件 `flashinfer_gqa.py` 与 `trtllm_attention_gqa.py`。
- 验证计划:用覆盖 flashinfer 与 trtllm api 后端的既有测试,并在三个模型上跑 GSM8K:Llama 3.3 70B、GPT OSS、nvidia/Qwen3.5-397B-A17B-NVFP4;在 SM90 与 SM103 上测试 Spec-decode、DCP、FP8 KV Cache、NVFP4 KV Cache 特性以保证兼容。
- 作者本地测试后倾向把 (1) 与 (3) 合并为单个 PR。

## 优化点与收益
正文未给出量化数字。收益为工程层面:后端文件与职责拆分后,降低 wrapper 内多 backend 交织带来的复杂度,使新特性(如 flashinfer_gqa api、DCP、spec decode、NVFP4 KV 路径)更容易添加与测试。

## 现状与落地
状态 open,处于 RFC 讨论阶段;拆分 PR #47786 已在正文中列为第一阶段实施载体。

## 关联
- https://github.com/vllm-project/vllm/pull/47786(拆分 PR)
- [标准 attention 支持矩阵文档](https://github.com/vllm-project/vllm/blob/main/docs/design/attention_backends.md#standard-attention-mha-mqa-gqa-backends)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
