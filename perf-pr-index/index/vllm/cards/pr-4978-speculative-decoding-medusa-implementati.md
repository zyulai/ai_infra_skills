# PR #4978: [Speculative Decoding] Medusa Implementation with Top-1 proposer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @abhigoyal1997 |
| 合并 | 2024-07-10 |
| 分类 | Speculative Decoding（tier: major） |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/4978 |

## 一句话结论

Medusa top-1 投机解码在 H100/MT-Bench 小 batch 提升吞吐，Mistral-7B bs1 128.73→155.06 tok/s。

## 动机

PR 正文说明本 PR 实现 Medusa 方法，用 heads 的 top-1 预测生成 speculations，并标注 FIX #1023、FIX #4669、Fix FasterDecoding/Medusa#41。作者指出配合 tree-style speculation 与 verification 应能带来更高提升，Medusa heads 用一组公开指令训练、正计划发布到 Huggingface Hub。PR 正文未详述改动前的性能瓶颈或更细的算法设计。

## 优化点

正文只说明实现 Medusa approach、由各 head 的 top-1 预测生成 speculations，并明确 tree-style speculation 与 verification 会带来更高提升（本 PR 为 top-1 proposer，未含 tree-style）。从 diff 文件看改动为：新增 Medusa 模型实现 `vllm/model_executor/models/medusa.py` 与 Medusa spec decode worker `vllm/spec_decode/medusa_worker.py` 并接到 `vllm/spec_decode/spec_decode_worker.py`；注册 `MedusaConfig`（`vllm/transformers_utils/configs/medusa.py` 及 `config.py`、`configs/__init__.py`）；在 `vllm/model_executor/models/__init__.py` 与 `vllm/worker/worker.py` 接入；新增 e2e 测试 `tests/spec_decode/e2e/test_medusa_correctness.py`。PR 正文未注明更细的内核或调度机制。

## 关键改动

```text
vllm/model_executor/models/medusa.py: 新增 Medusa 模型实现
vllm/spec_decode/medusa_worker.py: 新增 Medusa top-1 proposer 的 spec decode worker
vllm/spec_decode/spec_decode_worker.py: 接入 Medusa spec decode 路径
vllm/transformers_utils/configs/medusa.py: 新增 MedusaConfig 并在 config.py / configs/__init__.py 注册
tests/spec_decode/e2e/test_medusa_correctness.py: 新增基于 greedy equality 的 e2e 正确性测试
```
```text
# main model
# lmsys/vicuna-7b-v1.3 was to be used but it's causing
# OOM in CI pipeline, so using a smaller model.
MAIN_MODEL = "JackFram/llama-68m"

# speculative model
SPEC_MODEL = "abhigoyal/vllm-medusa-llama-68m-random"
```

## 性能声明

MT-Bench 上的 total tokens generated/sec（模型 Mistral-7B-Instruct-v0.2 与 Meta-Llama-3-8B-Instruct，硬件 H100，温度 0/1，concurrency 64/32/16/8/4/2/1）。小 batch 提升：Mistral-7B T=0 bs1 128.73 → +medusa 155.06、bs2 245.14 → 293.14、bs4 453.78 → 550.85、bs8 792.05 → 911.77；T=1 bs1 126.73 → 164.38；Meta-Llama-3-8B T=0 bs1 121.27 → 141.33、T=1 bs1 121.05 → 147.08。正文结论 "So for smaller batch sizes, we see improvement in tokens generated per sec."；高并发反而下降，例如 Mistral-7B T=0 bs64 2425.03 → 1769.62、Meta-Llama-3-8B T=0 bs64 2275.64 → 1533.34。测试条件：模型见上、硬件 H100、workload MT-Bench、指标 total tokens generated/sec、温度 0/1、concurrency 1-64；PR 未注明精度。

## 验证与回归风险

新增 `tests/spec_decode/e2e/test_medusa_correctness.py`，基于 greedy equality（投机解码输出与非投机解码逐序列一致，依据 rejection sampling 保证目标模型输出分布一致），覆盖 batch size 1、>1、preemption、不同 speculative tokens 数；配置 JackFram/llama-68m 作主模型、abhigoyal/vllm-medusa-llama-68m-random 作 speculator，float32，MAX_SPEC_TOKENS=5、enforce_eager=True、use_v2_block_manager=True、output_len 128。回归风险：正文数据显示高并发（bs32/64）吞吐低于非 Medusa 基线；tree-style speculation 与 verification 本 PR 未实现（作者称会带来更高提升）；依赖额外训练的 Medusa heads 权重。

## 关联

PR 正文标注 FIX #1023、FIX #4669 与 Fix FasterDecoding/Medusa#41；未给出同区域其他优化 PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
