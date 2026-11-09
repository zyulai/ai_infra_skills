# PR #57250: [Core] structured generation mode for DiffusionGemma model (Jev-like)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mmastrac |
| 合并 | 2026-09-22 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, speculative-decoding, ready, mrv2, scheduler |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,含约 6 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/57250 |

## 一句话结论

为 DiffusionGemma 增加 Jev 式结构化读出模式:单次前向输出整张 canvas 的多项决策,吞吐达约 162 decisions/s。

## 动机

DiffusionGemma 这类 diffusion LM 已具备构造 Jev-like 模型的全部组件:在结构化输出的固定 canvas 槽位上,模型可一次给出带 logprobs(可推导 entropy)的答案,用于表达置信度;entropy 超阈值时再多次采样读出以估计真实置信。此前 vLLM 缺少预置(seed)canvas、只读请求、步数限制等机制,无法把该模型用于这类"有界选项"结构化任务;另有并发 DiffusionGemma 请求在 logprobs stash 上互相踩踏导致 HTTP 500 的 bug。

## 优化点

核心机制是给 diffusion 采样路径引入四个由请求 `vllm_xargs` 驱动的开关:`diffusion_seed_canvas` 在 prefill 后用固定 token 替换随机 canvas;`diffusion_pinned` 用 pin mask 让这些位置在每步 denoise 保持 seed 值,并把其 self-conditioning soft embed 置零;`diffusion_read_only` 让请求在收敛步直接 emit argmax canvas 并跳过 commit 前向(logprobs 取温度 1);`diffusion_max_steps` 限制 denoise 步数。同时按请求引入 per-slot canvas 宽度,调度器只调度该宽度的 draft token;只跑一步的请求不再消费 soft embed,编译核里据此跳过对整个 vocab 矩阵的 probs@embed matmul。另修复并发请求 stash 丢失:新增 `_concat_logprob_stashes` 把不同收敛步、不同宽度的 logprobs stash 拼接(pad 到最大宽度)。

## 关键改动

```text
vllm/model_executor/models/diffusion_gemma.py: 核心。_compiled_sample_step 增加 pin_mask/seed_canvas/read_only/max_steps/compute_sc 输入,支持 pinned 位置保 seed、read-only 收敛步 emit、跳过 SC matmul;新增 _concat_logprob_stashes 修复并发 stash 丢失;slot 状态新增 seed/pin/read_only/canvas_width
vllm/config/vllm.py: diffusion_config + async_scheduling 时自动切换到 DiffusionAsyncScheduler
vllm/config/scheduler.py: 对 vllm 内部 scheduler 类名不再打"性能降级"警告
vllm/v1/core/sched/diffusion_scheduler.py: 新增 DiffusionAsyncScheduler 与 diffusion_canvas_width(diff 中经测试引用)
vllm/model_executor/models/config.py: use_non_causal=True 的说明改为"默认后端顺序偏好 FlashAttention"
```

关键代码(read-only emit 逻辑):

```python
emit = is_commit | (is_denoise & converged & read_only[decode_slots])
sampled[decode_idx] = argmax_canvas[decode_slots].to(sampled.dtype) * emit[:, None]
num_sampled[decode_idx] = (emit * valid_canvas_len).to(num_sampled.dtype)
```

## 性能声明

PR 正文在 1 台 DGX Spark(单 GPU 进程)上、`nvidia/diffusiongemma-26B-A4B-it-NVFP4`、32 行 canvas、每请求 3 个决策、schema 前缀命中缓存的条件下测得:单并发 8.7 req/s(0.12s),32 并发 54.0 req/s(0.58s),约 162 decisions/s。各类只读任务单次读出延迟约 292-530ms;分类任务中 auto 策略只在首次读不确定(H1>0.1)时才重读(最多 4 次)。标注用途为 prototype 示例服务器 `/v1/systemone`。

## 验证与回归风险

新增 `tests/v1/sample/test_diffusion_gemma_reads.py`,并在 `tests/v1/core/test_scheduler.py` 增加 canvas 宽度默认值、调度器行为等用例;测试命令 `pytest tests/test_sampling_params.py tests/v1/sample/test_diffusion_gemma_reads.py`。风险面:改动了共享的 `_compiled_sample_step` 与 slot 状态管理,影响 DiffusionGemma 普通生成路径(dynamo recompile limit 提到 64、eager 回退时的 dtype cast);scheduler 警告逻辑改动影响所有自定义 scheduler 用户;`use_non_causal` 说明变更关联后端自动选择。

## 关联

前置拆分 PR:#57414(logprobs stash 只在 commit 步发放)、#57416(prefill-only batch 按 model state 行数给 logit)、#57417(converging 步支持 logprob_token_ids)、#57462、#57589(多模态修复)。同区域:DiffusionGemma 相关 scheduler/模型优化(同分类 Model-specific Fast Path 其他 PR 见索引)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
