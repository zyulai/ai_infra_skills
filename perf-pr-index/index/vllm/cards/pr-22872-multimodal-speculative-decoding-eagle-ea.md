# PR #22872: [Multimodal][Speculative Decoding]Eagle Eagle3 mm support, enablement on qwen2.5vl

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @david6666666 |
| 合并 | 2025-09-27 |
| 分类 | Speculative Decoding (tier: major) |
| Label | performance, new-model, speculative-decoding, ready, v1, llama, qwen |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);改动集中在 `vllm/v1/spec_decode/eagle.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/model_executor/models/llama_eagle3.py` 等 |
| 链接 | https://github.com/vllm-project/vllm/pull/22872 |

## 一句话结论

让 EAGLE/EAGLE3 投机解码支持多模态(M-RoPE)目标模型,并在 Qwen2.5-VL-7B 上实测 e2e 提速 1.3x。

## 动机

此前 vLLM 的 Eagle 投机解码在 `propose_draft_token_ids` 中对 M-RoPE 位置只有 TODO 注释(`# TODO(woosuk): Support M-RoPE.`),`Eagle3LlamaForCausalLM` 遇到 `inputs_embeds` 直接抛 `NotImplementedError`("does not support multimodal inputs yet"),多模态目标模型无法搭配 EAGLE3 drafter。这是 #20788 的后续,目标是在 Qwen2.5-VL 上启用 EAGLE3(采用 Rayzl/qwen2.5-vl-7b-eagle3-sgl drafter)。

## 优化点

两条主线。一是 M-RoPE 位置支持:`EagleProposer` 按 `uses_mrope` 分配 `(3, max_num_tokens)` 的 `mrope_positions` 缓冲,propose 各阶段(位置拷贝、`positions += 1`、越界 clamp、slot mapping 计算)均按 M-RoPE 的第 0 维处理;`gpu_model_runner` 同样通过 `_get_positions` 抽象区分 mrope/普通位置缓冲。二是 multimodal 输入支持:`LlamaModel`(eagle3 drafter)新增 `get_input_embeddings` 并接受外部 `input_embeds` 而不再抛错;`Qwen2_5_VLForConditionalGeneration` 实现 `SupportsEagle3`,提供 `set_aux_hidden_state_layers`/`get_eagle3_aux_hidden_state_layers`(取第 2、`num_layers//2`、`num_layers-3` 层作为 aux hidden states)。另为 `vllm bench serve` 新增 MMStar 多模态数据集,便于基准测试。

## 关键改动

```text
vllm/v1/spec_decode/eagle.py: mrope_positions 缓冲与 _get/_set_positions、
  M-RoPE 越界 clamp/slot mapping、Qwen2.5-VL image_token_id 适配
vllm/v1/worker/gpu_model_runner.py: _get_positions 抽象,替换 M-RoPE TODO
vllm/model_executor/models/llama_eagle3.py: drafter 接受 inputs_embeds
vllm/model_executor/models/qwen2_5_vl.py: 实现 SupportsEagle3 与 aux 层配置
vllm/benchmarks/datasets.py: 新增 MMStarDataset
```

关键片段(EagleProposer 位置缓冲):

```python
self.uses_mrope = self.vllm_config.model_config.uses_mrope
if self.uses_mrope:
    # M-RoPE need (3, max_num_tokens)
    self.mrope_positions = torch.zeros((3, self.max_num_tokens), ...)
else:
    self.positions = torch.zeros(self.max_num_tokens, ...)
```

## 性能声明

PR 正文:e2e speed up 1.3x。条件:`vllm serve Qwen2.5-VL-7B-Instruct`,bfloat16,TP=1,max-num-seqs 128,max-model-len 8192,关闭 prefix caching,eagle3 投机配置(num_speculative_tokens 3,draft TP=1);benchmark 用 `vllm bench serve`,数据集 Lin-Chen/MMStar(val),50 prompts,max-concurrency 2,temperature 0。eagle3:吞吐 1.56 req/s、输出 124.09 tok/s、TPOT 14.07 ms;无 eagle3:1.12 req/s、94.66 tok/s、TPOT 20.48 ms。

## 验证与回归风险

`tests/v1/e2e/test_spec_decode.py` 新增 qwen2_5_vl_eagle3 用例,但被标记 skip(head_dim 不是 32 的倍数);`tests/models/registry.py` 注册 `Eagle3Qwen2_5vlForCausalLM` 示例检查。风险:M-RoPE 分支改变所有 mm/eagle 位置处理路径,slot mapping 与 clamp 逻辑在 mrope/非 mrope 间重复,易出错;`LlamaModel` 去掉 `@support_torch_compile` 装饰器,可能影响 drafter 的 compile 优化路径。

## 关联

前序:#20788(Eagle Eagle3 mm support 初始工作)。同分类其他 PR:#24505、#25916、#5765、#13365(EAGLE 系列相关)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
