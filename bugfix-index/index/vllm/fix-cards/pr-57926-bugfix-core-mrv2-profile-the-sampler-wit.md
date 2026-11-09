# PR #57926: [Bugfix][Core][MRV2] Profile the sampler with the params the warm-up actually uses

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57926](https://github.com/vllm-project/vllm/pull/57926) |
| 作者 | longluubao |
| 合并日期 | 2026-10-26 |
| labels | bug, ready, mrv2 |
| 热度 | 👍 0 · 💬 34 |
| 分类 | Sampler / Detokenization |
| bug 类型 | general |

## 一句话结论

MRv2 内存剖析的 dummy 批不带采样参数,sampler 的 fp32 logits 开销未计入峰值,KV cache 预算偏大,warm-up 阶段贴近 OOM;剖析改用 warm-up 同款参数。

## 问题现象

`profile_run` 用 `InputBatch.make_dummy` 构建 dummy 批,但不注册任何采样参数,于是每个 dummy 请求都令 `Sampler.needs_logits_processing` 为 False,`apply_sampling_params` 在 `if not np.any(self.needs_logits_processing[...])` 处提前返回——fp32 logits 上转型及其背后的 logit-bias、penalty、bad-words 工作全部不进入被剖析的内存峰值,KV cache 据此按一个比真实便宜的 sampler 定容。而 `warmup_kernels` 在 KV cache 分配之后才运行,用的是会真正触发这些参数的 `SamplingParams.for_sampler_warmup()`,第一次走到昂贵路径时已无内存可还,可能在该处 OOM。作者在 vLLM 0.29.0 的 L4 上记录过硬性失败: `warmup.py` line 348 → `sampler.py` line 222 `logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)` 处 `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 150.00 MiB`;并如实说明对当前 main 重跑未能复现硬失败(仅在 `enforce_eager=True` 下出现可恢复的非致命 CUDACachingAllocator OOM)。另在 8x RTX PRO 4500(32GB, SM120)、gpt-oss-120b MXFP4 TP=8 上独立复现同类问题:0.95 利用率出问题、0.90 消失。

## 根因

正文明确披露:内存剖析与 warm-up 使用两套不同的采样参数,前者(`make_dummy` 注册空参数)使 `apply_sampling_params` 早退,fp32 logits upcast 等开销从未进入剖析峰值;`_dummy_sampler_run` 中既有的 `NOTE(woosuk)` 注释记录的正是这个缺口。结果是 profiler 低报真实成本,启动在利用率高位附近变得脆弱。

## 修复方案

改 `vllm/v1/worker/gpu/model_runner.py` 的 `_dummy_sampler_run`(仅 MRv2):先跑一次默认 pass(模型 dtype logits,编译真实首步用到的 kernel);非 CPU 平台上,再为每个 dummy 请求注册 `SamplingParams.for_sampler_warmup()`,`apply_staged_writes()` 后再跑一次 sampler,使剖析峰值覆盖 fp32 pipeline:

```diff
+        warmup_params = SamplingParams.for_sampler_warmup()
+        for req_idx in range(num_reqs):
+            self.sampler.add_request(req_idx, warmup_params)
+        self.sampler.apply_staged_writes()
         self.sampler(logits, dummy_input_batch)
```

CPU worker 因按进程 RSS 定容且无 warm-up OOM 可防,显式提前返回。作者刻意不做状态恢复(说明理由:恢复发生在 sampler 调用之后不影响峰值,且 warm-up 随后会重建同样的残留)。新测试在未打补丁时 3 failed、打补丁后 3 passed。代价是同利用率下 KV cache 容量约降 1.3%(0.98: 1,733,440→1,710,928 tokens;0.97: 1,714,176→1,691,680)。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;问题仅存在于 MRv2(`vllm/v1/worker/gpu/model_runner.py`),MRv1 不在本次范围。
- 回归面:按 diff,只改剖析路径,不改变采样行为与模型输出;用户可见的变化是同 `gpu_memory_utilization` 下 KV cache 减少约 1.3%。多出的 sampler pass 会增加启动剖析耗时;若未来该路径增长,两个 pass 保证剖析与 warm-up 同步。测试 813+41+170 等全部通过,pre-commit 通过。

## 关联

- https://github.com/vllm-project/vllm/pull/43013 (标题最接近,但为 MRv1、意图相反)
- https://github.com/vllm-project/vllm/pull/51590 (互补:修正 CUDA graph 池预估)
- https://github.com/vllm-project/vllm/pull/46270 (投机解码范围,MRv1)
- https://github.com/vllm-project/vllm/pull/46883 (FlashInfer workspace 同类思路)
- https://github.com/vllm-project/vllm/pull/52054 (削减峰值,正交)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
