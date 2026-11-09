# PR #22169: [Model][V1] Support Ernie MTP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyxinyang |
| 合并 | 2025-08-20 |
| 分类 | Speculative Decoding |
| Label | documentation, performance, new-model, speculative-decoding, v1 等多标签 |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 6 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/22169 |

## 一句话结论

为 ERNIE-4.5 系列 MoE 模型接入 MTP 投机解码(method `ernie_mtp`),复用 EAGLE 式 V1 投机框架,目标 model 一个 MTP 层、每次投机 1 个 token。

## 动机

ERNIE-4.5-21B-A3B 等百度 MoE 模型自带 `num_nextn_predict_layers` 的 MTP 模块,架构与 DeepSeek MTP/GLM4-MoE MTP 同类,但 vLLM 此前不识别 `ernie4_5_moe`/`ernie_mtp` model_type,这些模型无法在 V1 引擎上使用投机解码加速。

## 优化点

新增 `vllm/model_executor/models/ernie_mtp.py`:`ErnieMultiTokenPredictorLayer` 由 `mtp_emb_norm`/`mtp_hidden_norm` 两个 RMSNorm、`mtp_linear_proj`(hidden_size*2→hidden_size 的拼接投影)和一个 `LlamaDecoderLayer` 组成,forward 把当前 token embedding 与上一层 hidden state 拼接投影后过 decoder block;`ErnieMTP` 复用目标模型的 embed_tokens/lm_head(tie_word_embeddings 时共享权重),`load_weights` 用 stacked_params_mapping(qkv/gate_up 合并)加 `_rewrite_spec_layer_name` 把 checkpoint 的 `model.mtp_*.0.` 命名重写到 `model.layers.<num_hidden_layers>.*`。配置层:`SpeculativeMethod` 增加 `ernie_mtp`,`hf_config_override` 把 `ernie4_5_moe` 重映射为 `ernie_mtp` 并注入 `n_predict`/`ErnieMTPModel` architecture,speculative config 的 draft model 自动取目标模型,`num_speculative_tokens>1` 时告警(仅一层 MTP);`get_layers_start_end_indices`、`use_eagle()`、worker 的 return_hidden_states 集合、`eagle.py` 的 propose 分支(deepseek_mtp 同款:直接返回 hidden states 而非二元组)都把 ernie_mtp 纳入。用法:`--speculative-config '{"method": "ernie_mtp", "num_speculative_tokens": 1}'`。

## 关键改动

```text
vllm/model_executor/models/ernie_mtp.py: 新增 ErnieMTP draft 模型(emb_norm+hidden_norm+linear_proj+LlamaDecoderLayer)
vllm/config/__init__.py: SpeculativeMethod/hf_config_override/spec config 逻辑接入 ernie_mtp
vllm/v1/spec_decode/eagle.py: propose 按 deepseek_mtp 同样方式处理 ernie_mtp 的 hidden states
vllm/worker/worker.py: target/draft 模型 return_hidden_states 白名单加入 ernie_mtp
tests/models/registry.py: 注册 baidu/ERNIE-4.5-21B-A3B-PT 测试条目
```

## 性能声明

PR 正文 Test Result 仅附两张截图(数字在图片中,素材文本未提取到具体值);测试配置为 V1 引擎、bfloat16、ERNIE-4.5-21B-A3B-PT、`num_speculative_tokens: 1`。PR 未注明可引用的量化加速数字与硬件。

## 验证与回归风险

- 依赖 `tests/models/registry.py` 的模型注册测试(speculative_model 指向自身)与既有 spec-decode CI;未见 kernel 级新测试。
- 风险:仅支持单层 MTP(`spec_step_idx == 0` assert);`mtp_linear_proj` 用 nn.Linear(非 parallel 层),TP 切分依赖 checkpoint 布局;权重加载对 `mtp` 命名重写较脆弱(expert 名跳过逻辑需要 checkpoint 侧名字完全匹配);对 deepseek_mtp 共享路径的改动(method 白名单、propose 分支)需回归 DeepSeek 投机解码。

## 关联

- 同区域:#12755(DeepSeek MTP)、#16937(EAGLE-3)、Glm4MoeMTP、mimo_mtp——同一 MTP/EAGLE 家族;#16937/#14702 构成 V1 spec-decode 基础设施。
- Ernie 模型本体支持 PR 为其前序(ernie4_5 架构接入)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
