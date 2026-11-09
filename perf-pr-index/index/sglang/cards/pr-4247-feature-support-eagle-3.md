# PR #4247: [Feature] Support EAGLE 3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @chromecast56 |
| 合并 | 2025-03-18 |
| 分类 | Speculative Decoding (tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files) |
| 链接 | https://github.com/sgl-project/sglang/pull/4247 |

## 一句话结论

为 SGLang 增加 EAGLE-3 投机解码:目标模型捕获低/中层辅助 hidden states,草稿模型用 3 倍拼接特征 + token map 做树形草稿。

## 动机

添加 EAGLE-3(arXiv:2503.01840)支持。与 EAGLE-2 相比,EAGLE-3 移除 feature prediction 目标、引入低层与中层特征、并以 on-policy 方式训练(描述来自 PR 内文档改动),草稿接受率更高,因此需要运行时把目标模型多个中间层的 hidden states 一并喂给草稿模型。

## 优化点

机制上分三块:(1) 目标模型侧,`llama.py` 增加 `layers_to_capture = [2, num_layers//2, num_layers-3]`,forward 逐层收集 `hidden_states + residual` 作为辅助特征;`logits_processor` 接收 `aux_hidden_states` 并在 capture_hidden_mode 下把低/中/高三层特征沿最后一维 cat 成 `3 * hidden_size` 的向量存入。(2) 草稿模型侧,新增 `llama_eagle3.py`:单层 draft(强制 `num_hidden_layers == 1`),用 `fc` 把 3*hidden_size 投影回 hidden_size,decoder layer 中把 token embedding 与 hidden state 各自 norm 后 cat 成 `2*hidden_size` 作为 qkv 输入;支持 untied LM head 与 `d2t` hot token id(d2t 存 draft 词表 id 到 target 词表 id 的差)。(3) 调度侧,`SpeculativeAlgorithm` 增加 `EAGLE3` 枚举并复用 `is_eagle` 路径;`eagle_worker` 对 EAGLE3 只共享 embedding 不共享 lm_head,hot token id 从草稿权重读取而忽略外部 token map;cuda_graph_runner 为 EAGLE3 分配 3 倍宽的 hidden_states buffer。

## 关键改动

```text
python/sglang/srt/models/llama_eagle3.py: 新文件,EAGLE-3 单层草稿模型(fc 投影 + embed/hidden cat 输入)
python/sglang/srt/models/llama.py: layers_to_capture 捕获第 2/中层/倒数第 3 层 aux hidden states
python/sglang/srt/layers/logits_processor.py: forward 接收 aux_hidden_states 并 cat 成 3x hidden_size 存储向量
python/sglang/srt/speculative/eagle_worker.py: EAGLE3 不共享 lm_head,从权重读 d2t hot token id
python/sglang/srt/server_args.py: --speculative-algorithm 增加 EAGLE3 选项
```

## 性能声明

PR 正文未提供 benchmark 数字,PR 未注明测试条件(仅引用 EAGLE-3 论文链接)。

## 验证与回归风险

新增 `TestEAGLE3Engine`(Llama-3.1-8B-Instruct + jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B,num_steps=5,topk=16,num_draft_tokens=64)与 speculative_decoding.ipynb 文档示例。风险:`LlamaForCausalLM.forward` 签名改为可能返回 `(hidden_states, aux_hidden_states)` 元组,影响所有 Llama 系模型路径;EAGLE3 复用 `disable_overlap_schedule=True` 与 `max_running_requests=32` 的既有 EAGLE 约束。

## 关联

前序:EAGLE/EAGLE-2 支持;后续:#24663(support newer EAGLE-3 drafters)、#11398(spec overlap for EAGLE)、#6606(EAGLE draft extend CUDA graph)。
