# PR #51953: [Bugfix][MRV2] Fix DiffusionGemma runtime OOM via tiled logits projection

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51953](https://github.com/vllm-project/vllm/pull/51953) |
| 作者 | guan404ming |
| 合并日期 | 2026-10-30 |
| labels | bug, mrv2, ready |
| 热度 | 👍 2 · 💬 10 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
DiffusionGemma 采样器对全 batch logits 做投影,显存随并发增长,首个 decode batch 即 OOM;改为逐 tile 投影 logits 并在 profile_run 中预留瞬态显存。

## 问题现象
Fixes #50699:全 batch logits 随并发增长,首个 decode batch 直接 OOM。作者给出 A/B 实测(RTX 5090 32GB,NVFP4 26B,8 并发 x 512 token):`gpu-memory-utilization` 0.85 时 engine 在首个 batch 死亡(`Tried to allocate 2.00 GiB`,即 #50699 的失败),0.90 时启动即 OOM;修复后 0.85/0.90 均可服务,0.70-0.80 照常,0.92 仍有启动 OOM(prompt-logprobs 路径,属后续工作)。最大稳定 KV cache 从 508k 提升到 689k token(+36%),低利用率下预留机制约占 55k token。

## 根因
正文披露:full-batch logits 随并发增长并 OOM 首个 decode batch——即采样器在投影 logits 时一次性物化整个 batch 的 `[rows, vocab]` 张量,规模随并发请求量线性放大,峰值显存远超 profile 阶段的预留。正文并澄清与 #46155 的区别:那是 opt-in 环境变量方案,本 PR 修复的是默认路径,非重复。

## 修复方案
1. `vllm/model_executor/models/diffusion_gemma.py`:`DiffusionSampler.__call__` 从接收 `logits` 改为接收 `hidden_states`,logits 投影移入每个 tile 内部(`tile_logits = self.compute_logits(tile_hidden)`);`get_top_k_top_p`/`apply_top_k_top_p` 与约束解码的 `_mask_rows_to_allowed` 均改为对 tile_logits 操作(phantom 行用 masked_fill 置 0,防止 top_k/top_p 的 -inf 把 phantom 行变成 NaN);显存预算因子由 `* 3` 调到 `* 5`(新增投影 fp32 logits 及其 top_k/top_p 或 allowed-set 副本两份瞬态)。新增 `compute_sample_logits`(返回 hidden_states,即 logits 在采样器内计算)与 `dummy_sampler_run`/`profile_run` hook:profile_run 用 1 请求、canvas 全宽的 dummy batch 跑一遍 decode 管线,让 profiling 阶段把 tile 瞬态显存计入预留。
2. `vllm/v1/worker/gpu/model_runner.py`:`_dummy_sampler_run` 先询问 `dummy_sampler_run` hook(返回 True 则跳过默认路径);`sample` 中 logits 计算改为 `self.model_state.compute_sample_logits(sample_hidden_states)`。
3. `vllm/v1/worker/gpu/model_states/interface.py`:接口新增 `compute_sample_logits` 默认实现(仍是 `model.compute_logits`)与 `dummy_sampler_run`(默认 False),不影响非 diffusion 模型。
4. 新增 `tests/v1/worker/test_gpu_diffusion_sampler.py`:验证 group=1 分 tile 与全 batch 结果一致、强制分 tile 时峰值显存低于全 batch 一半、profile_run 后状态正确复位。

## 影响与回归风险
- 影响窗口:正文未说明(MRV2/DiffusionGemma 路径)。
- 回归面:DiffusionGemma 采样与 GPU worker 的 dummy/profiling 流程。正文注明 tiling 后输出不再是 bit-identical(浮点求和顺序变化),GSM8K 200 题 5-shot 92.0%→92.5%(视为噪声),吞吐 5.67→5.58 q/s;非 diffusion 模型走接口默认实现,行为不变;0.92 高利用率下的启动 OOM 仍未解决(prompt-logprobs 路径,作者列为 follow-up)。

## 关联
- #50699(Fixes)
- #46155(区分:opt-in 环境变量方案,非重复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
