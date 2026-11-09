# PR #12755: [Model][Speculative Decoding] DeepSeek MTP spec decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @luccafong |
| 合并 | 2025-02-19 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, ready, ci/build |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,约 14 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/12755 |

## 一句话结论

为 DeepSeek V3/R1 实现 MTP(next-n prediction)投机解码:接受率 81-82.3%,QPS=1 时 TPOT 加速 1.63x。

## 动机

实现 issue #12181:DeepSeek V3/R1 checkpoint 自带 MTP(multi-token prediction)层,可在推理时复用为 draft 模型做 next-n prediction,无需额外训练 draft。此前 vLLM 无法利用这些权重,DeepSeek 系列只能自回归解码。当前实现仅在 MTP k=1 模型上验证过。

## 优化点

新增 `DeepSeekMTPModel`:MTP 层是挂在目标模型之后的 `DeepseekV2DecoderLayer`,输入由 token embedding(经 enorm 归一,position 0 处屏蔽)与上一隐状态(经 hnorm)拼接、`eh_proj` 投影后送入该 block,shared head(RMSNorm + ParallelLMHead)输出 logits。draft 直接复用目标模型同一 checkpoint 的 MTP 权重:`hf_config_override` 把 `deepseek_v3` 的 model_type 改写为 `deepseek_mtp` 并写入 `n_predict`/`DeepSeekMTPModel` 架构;只要用户给 `num_speculative_tokens` 就自动用目标模型自身做 draft。配置侧把 `deepseek_mtp` 纳入 `is_deepseek_mla` 判定、按 `num_nextn_predict_layers` 计算 PP 层区间。加载时按 `get_spec_layer_idx_from_weight_name` 提取 MTP 层权重,gate/up 合并为 gate_up_proj,专家权重走 FusedMoE mapping。draft TP 可与目标 TP 不同(`draft_tensor_parallel_size` 可选,默认同 target)。

## 关键改动

```text
vllm/model_executor/models/deepseek_mtp.py: 新增 DeepSeekMTP/DeepSeekMultiTokenPredictor(Layer)/SharedHead 与权重加载
vllm/config.py: hf_config_override(deepseek_v3→deepseek_mtp)、maybe_create_spec_config 支持 num_speculative_tokens 自动选 draft、is_deepseek_mla/层区间判定
vllm/spec_decode/draft_model_runner.py, spec_decode_worker.py: V0 spec decode worker 支持 MTP draft(传 previous_hidden_states/spec_step_idx)
vllm/worker/model_runner.py, model_runner_base.py, worker.py, worker_base.py: broadcast draft 输入/draft tp 支持等配套改动
vllm/model_executor/models/registry.py: 注册 DeepSeekMTPModel
```

关键代码(MTP 层前向的核心拼接):

```python
inputs_embeds[positions == 0] = 0
inputs_embeds = self.enorm(inputs_embeds)
previous_hidden_states = self.hnorm(previous_hidden_states)
hidden_states = self.eh_proj(
    torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
hidden_states, residual = self.mtp_block(...)
```

## 性能声明

DeepSeek-R1,k=1(仅支持 1),TPOT 基准(单位 ms,PR 未注明硬件/精度):接受率 81%-82.3%。Draft TP=1、tensor-parallel-size 8:QPS 1/2/4/6/8 加速 1.63x/1.18x/1.26x/1.14x/1.0x(基线 TPOT 55.47→33.99 等)。Draft TP=8:QPS 1/2/4 加速 1.69x/1.32x/1.22x,QPS 6/8 小于 1.0。各种并发(Draft TP=8):并发 1/2/4/8/16/32 加速 1.34x/1.64x/1.28x/1.11x/1.22x/1.25x。结论:QPS=1 时 1.63x,QPS<8 有一定提升,高 QPS 无收益。

## 验证与回归风险

新增 `tests/spec_decode/e2e/test_mtp_correctness.py`;registry 添加随机权重小模型 `luccafong/deepseek_mtp_main_random`(draft `deepseek_mtp_draft_random`)供 CI;.buildkite/test-pipeline.yaml 更新。风险面:V0 spec decode 路径大面积改动(model_runner/worker/spec_decode_worker),影响所有投机解码方法;config.py 的 MLA 判定与 PP 层区间逻辑影响 DeepSeek 系列非投机部署;draft TP 与 target TP 解耦是新行为。

## 关联

同区域:后续 Kimi K3 MTP(#50000,`kimi_k3_mtp` 同样的 hf_config_override 模式)、DSpark(#46995)、DFlash2(#52816);vLLM v1 引擎的 MTP 支持由后续 PR 演进(编号未注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
