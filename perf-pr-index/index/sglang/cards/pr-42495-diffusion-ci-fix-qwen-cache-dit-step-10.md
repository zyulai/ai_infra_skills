# PR #42495: [diffusion] CI: fix qwen cache-dit step 10/49 perf baselines

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @niehen6174 |
| 合并 | 2026-10-07 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci, diffusion |
| Diff | pack diffstat 缺失;diff 仅 h100.json 两个 step 基线值变更 |
| 链接 | https://github.com/sgl-project/sglang/pull/42495 |

## 一句话结论

修正 qwen cache-dit CI 逐步 perf 基线中 step 10/49 的假阳性——根因是无 device sync 的 host wall time 计量,并非性能回归。

## 动机

`qwen_image_t2i_cache_dit_enabled` 的 H100 per-step perf check 随机失败:09-30 至 10-04 期间 71% 的尝试超过 step-49 上限,该窗口内所有 final 1-GPU 失败都来自此例。#42546 的 debug 运行定位了原因:step 10 和 49 是 full step 之后的首个 cached step;per-step 时间是不带 device sync 的 host wall time;cache-dit residual check 是该步 host 首个阻塞点,于是这一步吸收了前一个 full step 排队的 GPU 工作。这是计量问题,不是回归。

## 优化点

纯基线修正:`perf_baselines/h100.json` 中 `qwen_image_t2i_cache_dit_enabled` 的 step 10/49 设为 CI 中位数 64.58 / 66.38 ms(原 48.7 / 45.11)。中位数来自 09-30 至 10-04 的 328 次 H100 CI 测量,正是该检查实际计量的 unsynced host wall time。其他基线与 tolerance 均不变;同时删除早前为 `ltx_2_5_diffusion_decoder_2gpus` 加入的 timing tolerance,其 host-bound slowdown 将由单独 PR 处理。

## 关键改动

```text
python/sglang/multimodal_gen/test/server/perf_baselines/h100.json: step 10: 48.7 → 64.58;step 49: 45.11 → 66.38
```

## 性能声明

PR 未声明新的性能优化,数字均为诊断数据(Debug run on CI H100,3 launches per arm,ms):

| Arm | Full step | Steps 10/49 | Other cached | E2E |
| --- | ---: | ---: | ---: | ---: |
| Fused epilogue on, unsynced(CI 现状) | 178 | 64 / 66 | 6.3 | 4310 |
| Fused off, unsynced | 196 | 55 / 56 | 6.5 | 4455 |
| Fused on, synced | 238 | 7.0 / 7.1 | 6.3 | — |
| Fused off, synced | 245 | 7.2 / 7.1 | 6.5 | — |

synced 计时下 step 10/49 约 7 ms,与其他 cached step 一致。另注明 #38584 的 fused QK-norm/RoPE epilogue 使 host launches 3031 → 2007,real GPU time 约快 3%,但 host enqueue 更快导致更多 GPU 工作溢入 step 10/49。

## 验证与回归风险

PR CI(fe3b48d)在相同 qwen 基线下通过。风险:基线锚定的是 unsynced host wall time,对 host enqueue 速率敏感;fused epilogue 等前端改动再提速 host 侧时,step 10/49 基线可能再次需要校准。`ltx_2_5_diffusion_decoder_2gpus` 的 tolerance 移除后,其 host-bound 慢化依赖后续 PR 解决。

## 关联

#42546(debug 运行,定位计量根因);#38584(Hopper fused QK-norm/RoPE epilogue,间接放大该假阳性)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
