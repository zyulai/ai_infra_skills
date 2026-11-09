# PR #23198: [diffusion] Fix --warmup-resolutions hang with --enable-cfg-parallel

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23198](https://github.com/sgl-project/sglang/pull/23198) |
| 作者 | mispa-ms |
| 合并日期 | 2026-04-23 |
| labels | npu, run-ci, diffusion |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论
warmup 合成的 Req 未启用 CFG,cfg-parallel 下 rank 1 返回 None 的 noise_pred 使 scheduler.step 崩溃,被 GPUWorker 吞掉后变成 30 分钟 gloo 静默超时;warmup Req 强制启用 CFG 并加请求侧校验护栏。

## 问题现象
以 `--enable-cfg-parallel --warmup --warmup-resolutions <list>` 启动 SGLang-diffusion 时出现 30 分钟的静默挂起。表面上是 `gloo` broadcast 超时(`dist_timeout=3600`,即 `Timed out waiting 1800000ms for recv operation to complete`)后 scheduler 终止,但真实故障更早:`scheduler.step()` 抛出

```
TypeError: unsupported operand type(s) for *: 'Tensor' and 'NoneType'
  at scheduling_flow_match_euler_discrete.py:537
```

该异常被 `GPUWorker` 捕获吞掉,`/health` 持续返回 200,直到真实请求到来才以 30 分钟广播超时的形式暴露——把 7 秒的逻辑错误拖成半小时的静默挂起。复现命令(正文照抄):`sglang serve --model-path Qwen/Qwen-Image-2512 --num-gpus 2 --enable-cfg-parallel --warmup --warmup-resolutions 512x512 1024x1024 --enable-torch-compile --trust-remote-code`。

## 根因
正文披露完整根因链:
1. `Scheduler.prepare_server_warmup_reqs`(scheduler.py:238-258)合成 warmup `Req(prompt="", negative_prompt="")`,未传 `sampling_params`;
2. `Req.__init__`(schedule_batch.py:169-182)在首次 setattr SamplingParams 字段时自动创建**基类** `SamplingParams()`(非模型特定子类),基类默认 `guidance_scale=1.0`;
3. `Req.validate`(schedule_batch.py:270-285)仅在 `cfg_scale > 1.0 且 negative_prompt is not None` 时才置 `do_classifier_free_guidance=True`,于是 cfg_scale=1.0 时该标志保持 False;
4. `_predict_noise_with_cfg`(denoising.py:1596-1621)在 `do_cfg=False` 时对 `cfg_rank != 0` 返回 `noise_pred_cond=None`;
5. `scheduler.step(model_output=None)` 在 `prev_sample = sample + dt * model_output` 处崩溃。

## 修复方案
三个 commit:
1. **scheduler 修复 warmup Req 合成**:`prepare_server_warmup_reqs` 改为条件 kwargs 构建;`enable_cfg_parallel` 开启时设置 `negative_prompt=DEFAULT_PLACEHOLDER_PROMPT`(非空真实词 "warmup",保证各 tokenizer 后端产生可预测的非退化 token 序列)与 `do_classifier_free_guidance=True`,使 rank 1 产出有效的 `noise_pred_uncond`、`_combine_cfg_parallel` 的 all-reduce 成功;cfg-parallel 关闭时 Req 构造与修复前逐字节一致:

```diff
+                if self.server_args.enable_cfg_parallel:
+                    req_kwargs["negative_prompt"] = DEFAULT_PLACEHOLDER_PROMPT
+                    req_kwargs["do_classifier_free_guidance"] = True
                 req = Req(**req_kwargs)
```

2. **请求侧防御护栏**:`InputValidationStage.forward` 在服务器以 `--enable-cfg-parallel` 启动但请求 `do_classifier_free_guidance=False` 时抛 `ValueError`,报错信息区分 negative_prompt 三种状态(not set/empty/set),并提示 `negative_prompt_embeds` 也可启用 CFG;
3. **单元测试**:`test/unit/test_cfg_parallel_warmup.py` 四个 CPU-only 用例覆盖两个 commit(正例 + 回归护栏)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `python/sglang/multimodal_gen/` 下 scheduler、input_validation 与新增测试。按正文 Before/After:warmup + cfg-parallel 从 30 分钟挂起变为正常;无 CFG 的真实请求打到 cfg-parallel 服务器从 30 分钟挂起变为立刻 `ValueError`(拒绝服务,属行为收紧);带 CFG 请求与不带 cfg-parallel 的 warmup 行为不变。风险点:依赖"非 CFG 请求可被 cfg-parallel 服务器接受"的调用方现在会收到 ValueError。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
