# PR #16937: [V1][Spec Decode] EAGLE-3 Support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2025-04-25 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, ready, v1 |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 10 个文件：新增 `vllm/model_executor/models/llama_eagle3.py`，修改 `llama.py`、`llama_eagle.py`、`registry.py`、`v1/spec_decode/eagle.py`、`v1/worker/gpu_model_runner.py`、`v1/core/sched/scheduler.py`、`config.py`、`engine/arg_utils.py`、`examples/offline_inference/eagle.py` 及测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/16937 |

## 一句话结论

V1 引擎引入 EAGLE3 投机解码（目标模型返回中间层辅助 hidden states 拼接给 drafter）：Llama-3.1-8B + RTX 4090 mt_bench 输出吞吐 82.47→143.81 toks/s，mean acceptance length 2.48→3.57。

## 动机

EAGLE3 与此前 EAGLE 的关键差异是 drafter 消费目标模型中间层 hidden states 的拼接（而非仅最后一层输出），可显著提高接受率。要在 vLLM V1 中支持，必须让目标模型把中间层辅助 hidden states 一并返回——这是任何想支持 EAGLE3 的模型都需要做的接口改动。参考实现为 SafeAILab/EAGLE 的 cnets.py。

## 优化点

三块机制。其一，目标模型侧接口：`LlamaForCausalLM` 新增 `aux_hidden_state_layers`，forward 中在指定层索引处缓存 `hidden_states + residual`，EAGLE3 取层 `(2, num_layers//2, num_layers-3)`；`gpu_model_runner` 以 `use_aux_hidden_state_outputs` 区分输出元组，把各层辅助 hidden states 沿最后一维 `torch.cat` 后传给 drafter。其二，drafter 模型：新建 `Eagle3LlamaForCausalLM`（单 midlayer 结构）——`fc` 先把 3x 拼接 hidden states 投影到 hidden_size，与 input_embeds 拼成 `[input_embeds, projected_hidden_states]`，最终投影并入 `QKVParallelLinear`（输入 2x hidden_size）；EAGLE 层返回一对 hidden states（norm 后用于采样、norm 前传入下一轮投机）；lm_head 为 32k 的 draft 词表，`compute_logits` 经 `draft_id_to_target_id`（d2t）映射散布回目标词表（其余置 -inf），权重加载跳过 t2t、改写 d2t。其三，接线与校验：`speculative_config` 支持 `method="eagle3"` 与模型名 "eagle3-" 自动检测，`_verify_args` 限定 target 为 Llama 系；scheduler 为 eagle3 设置 lookahead tokens；EAGLE3 缺 embedding 时从目标模型加载 embed_tokens（EAGLE 则共享 lm_head）。

## 关键改动

```text
vllm/model_executor/models/llama_eagle3.py: Eagle3LlamaForCausalLM（fc+单 midlayer+d2t 词表映射）
vllm/model_executor/models/llama.py: aux_hidden_state_layers 与 (hidden, aux) 双输出
vllm/v1/worker/gpu_model_runner.py: 收集并 cat 辅助 hidden states 传给 drafter
vllm/config.py: eagle3 方法识别与 Llama-only 校验
```

```python
def get_eagle3_aux_hidden_state_layers(self) -> tuple[int]:
    num_layers = len(self.model.layers)
    return (2, num_layers // 2, num_layers - 3)
```

## 性能声明

测试条件：RTX 4090，meta-llama/Llama-3.1-8B-Instruct，mt_bench question.jsonl，`num_spec_tokens 7`、`max_num_seqs 1`、80 prompts，V1 引擎，argmax 采样。baseline 为 EAGLE（yuhuili/EAGLE-LLaMA3.1-Instruct-8B）：output 82.47 toks/s、mean acceptance length 2.48。本 PR EAGLE3（yuhuili/EAGLE3-LLaMA3.1-Instruct-8B）：output 143.81 toks/s、mean acceptance length 3.57（逐 token 条件接受率约 75% 且不随位置明显衰减：token1 0.75→token7 0.13）。作者另注明：profiling 显示 drafting 时间随投机 token 数逼近 decode 时间，CUDA graphs 与消除 EAGLE 代码中的同步可进一步显著提升（本 PR 未做）。

## 验证与回归风险

测试：`tests/v1/e2e/test_spec_decode.py` 的 `test_eagle_correctness` 参数化为 eagle/eagle3 双路径（对比原模型与投机模型输出一致性）；`tests/models/registry.py` 注册 Eagle3LlamaForCausalLM 示例。风险面：eagle3 仅支持 Llama 系目标模型（config 显式 raise）；目标模型 forward 签名变为可能返回元组（`_dummy_run`/PP 中间层路径同步适配）；chunked prefill 在 V0 与 EAGLE 不兼容（报错）；draft 32k 词表的 d2t 映射引入额外 logits scatter 开销。

## 关联

前序：V1 EAGLE 支持（`llama_eagle.py`，本 PR 在其上扩展）；接受率测试依赖当时已 revert 的 #16367 实验性 diff。索引内同区域后续：#25916（EAGLE3 preamble 修复）、#42143（EAGLE3 读取 norm before fc 修复）、#22872（多模态 EAGLE）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
