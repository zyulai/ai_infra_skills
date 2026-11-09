# PR #29594: [Multimodal][Speculative Decoding]Eagle3 mm support, enablement on qwen3vl

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @EanWang211123 |
| 合并 | 2025-11-28 |
| 分类 | Speculative Decoding(tier: major) |
| Label | new-model, speculative-decoding, ready, v1, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/29594 |

## 一句话结论

为 Qwen3-VL 打通 Eagle3 多模态投机解码，MMStar 上总 token 吞吐提速 1.66x。

## 动机

正文 Purpose 说明本 PR 承接 #20788，做「Eagle3 mm support, enablement on qwen3vl」：目标模型为 `Qwen/Qwen3-VL-8B-Instruct`，Eagle3 草稿头来自 `Rayzl/qwen2.5-vl-7b-eagle3-sgl`，正文给出了二者的 HuggingFace 链接，意图是把既有的 Eagle3 多模态投机解码能力扩展到 Qwen3-VL 系列。从 diff 看，此前 `vllm/v1/spec_decode/eagle.py` 的多模态分支只识别 `Qwen2_5_VLForConditionalGeneration`，Qwen3-VL 未接入 Eagle3 路径，也没有提供草稿头所需的 aux hidden states。

## 优化点

(1) `Qwen3VLForConditionalGeneration` 声明 `SupportsEagle3` 接口，新增 `set_aux_hidden_state_layers` / `get_eagle3_aux_hidden_state_layers`（返回 `(2, num_layers // 2, num_layers - 3)` 三层），`forward` 在指定层收集 `hidden_states + residual` 到 `aux_hidden_states`，存在时返回 `(hidden_states, aux_hidden_states)`，供 Eagle3 草稿头取 aux hidden states。(2) `vllm/model_executor/models/registry.py` 注册 `Eagle3Qwen3vlForCausalLM` 映射到 `llama_eagle3.Eagle3LlamaForCausalLM`。(3) `vllm/v1/spec_decode/eagle.py` 把多模态分支的判断由单一模型名扩为 `["Qwen2_5_VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"]` 列表，沿用 `image_token_index` 设置。(4) `tests/models/registry.py`、`tests/v1/e2e/test_spec_decode.py` 增加相应覆盖。

## 关键改动

```text
vllm/model_executor/models/qwen3_vl.py: 声明 SupportsEagle3、收集/返回 aux_hidden_states、提供三层索引
vllm/model_executor/models/registry.py: 注册 Eagle3Qwen3vlForCausalLM
vllm/v1/spec_decode/eagle.py: 多模态分支纳入 Qwen3VLForConditionalGeneration
tests/models/registry.py: 登记新示例模型
tests/v1/e2e/test_spec_decode.py: e2e 投机解码测试覆盖
```

关键片段（多模态分支扩展）：

```diff
         if supports_multimodal(target_model):
             # handle multimodality
-            if (
-                self.get_model_name(target_model)
-                == "Qwen2_5_VLForConditionalGeneration"
-            ):
+            if self.get_model_name(target_model) in [
+                "Qwen2_5_VLForConditionalGeneration",
+                "Qwen3VLForConditionalGeneration",
+            ]:
                 self.model.config.image_token_index = target_model.config.image_token_id
```

## 性能声明

正文给出「Total Token throughput speed up 1.66x」。测试条件：`Qwen3-VL-8B-Instruct`，`CUDA_VISIBLE_DEVICES=4,5`（2 卡）、`--tensor-parallel-size 2`、`--max-model-len 32768`、`--max-num-seqs 32`，Eagle3 `num_speculative_tokens: 3`；`vllm bench serve` 走 `/v1/chat/completions`，数据集 hf `Lin-Chen/MMStar`（hf-split val），50 prompts、max-concurrency 2、temperature 0、top-p 1.0、seed 123。两组 Serving Benchmark Result：带 eagle3 的 Total Token throughput 249.97 tok/s（Output 208.94 tok/s、req/s 1.67、Mean TTFT 118.64 ms、Mean TPOT 8.57 ms、Mean ITL 16.17 ms、duration 29.88 s、Peak output 131.00 tok/s、P99 TTFT 306.27 ms、P99 TPOT 10.23 ms、P99 ITL 69.85 ms）；不带 eagle3 为 180.68 tok/s（Output 150.86 tok/s、req/s 1.22、Mean TTFT 105.26 ms、Mean TPOT 12.33 ms、Mean ITL 12.24 ms、duration 41.12 s、Peak output 168.00 tok/s、P99 TTFT 224.84 ms、P99 TPOT 13.17 ms、P99 ITL 17.43 ms），两组均 50 successful / 0 failed。草稿头配置为 `/models/Qwen3-VL-8B-Instruct-Eagle3/`、`num_speculative_tokens: 3`，服务侧指定 `--max-model-len 32768`、`--max-num-seqs 32` 与 `VLLM_USE_V1=1`。

## 验证与回归风险

PR 正文的 Test Plan 一节为空，未注明单测或 CI；Test Result 仅给出上述 serving benchmark 输出，未附正确性指标。diff 含 `tests/models/registry.py` 与 `tests/v1/e2e/test_spec_decode.py` 改动。风险：aux hidden state 三层索引固定为 `(2, num_layers//2, num_layers-3)`，仅按 Qwen3-VL-8B-Instruct + Eagle3（k=3）验证；其他多模态模型、硬件与精度未在正文给出数据；eagle.py 的多模态分支改为列表匹配后需覆盖两类 VL 模型。

## 关联

#20788（正文「follow pr」指明的前序 Eagle3 mm 支持 PR）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
