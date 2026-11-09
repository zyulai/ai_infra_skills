# PR #19463: [Perf] Improve/Fix-regression for FA3 in High QPS regimes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19463](https://github.com/vllm-project/vllm/pull/19463) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-06-24 |
| labels | ready, ci/build, qwen |
| 热度 | 👍 5 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论

高 QPS 下 FA3 的 combine 阶段耗时过长造成回归;升级 vllm-flash-attn 固定 commit(含 flash-attention#70 修复),100/200 req/s 下 TTFT 大幅回落。

## 问题现象

高请求速率下 FA3 花费过多时间在 combine 阶段(源自 issue #18619 的发现)。基准设置:`vllm serve <model> --disable-log-requests --max-num-seqs 1024 --block-size 16 --max-num-batched-tokens 2048 --no-enable-prefix-caching`,用 `vllm bench serve --request-rate 1,5,50,100,200 --request-for-secs 5 --random-input-len 600 --random-output-len 125 --ignore-eos` 压测。Qwen2.5-VL-3B-Instruct 上 main→PR:请求率 100 时 Median TTFT 175.15→83.14 ms、Median TPOT 30.11→18.68 ms;请求率 200 时 TTFT 3329.13→1940.63 ms、吞吐 8117.78→10376.26 output tok/s。Llama-3.1-8b 改善较小:请求率 50 时 TTFT 139.83→97.19 ms,100/200 档 TTFT 略降(1704.81→1572.63、7010.04→6730.15 ms)。低速率(1/5 req/s)两模型基本持平。

## 根因

正文披露:问题在于 FA3(FlashAttention-3)的 combine 阶段在高 QPS 场景消耗过多时间,结论来自 https://github.com/vllm-project/vllm/issues/18619 ;具体修复机制在其关联的 vllm-flash-attn 仓库 PR(vllm-project/flash-attention#70)中描述,正文要求该 PR 必须先落地。

## 修复方案

vLLM 侧的实际改动是更新外部依赖固定版本:`cmake/external_projects/vllm_flash_attn.cmake` 中 vllm-flash-attn 的 `GIT_TAG` 由 `763ad155a1c826f71ff318f41edb1e4e5e376ddb` 升级为 `2c6bcfc0feb3d9d4a57b243fc159a68aa9933f5b`,从而带入 flash-attention#70 中对 combine 阶段的修复:

```diff
           GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
-          GIT_TAG 763ad155a1c826f71ff318f41edb1e4e5e376ddb
+          GIT_TAG 2c6bcfc0feb3d9d4a57b243fc159a68aa9933f5b
```

diff 中另有一个新增的 `test-qwen` 子模块条目(Subproject commit 34c31c0a...)。正确性验证:lm_eval gsm8k(Qwen2.5-VL-3B-Instruct, 5-shot)本 PR 与 main 的 flexible-extract/strict-match 精度完全一致(0.6831 / 0.5383);另跑了 flash-attention#70 的基准测试与 CI。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;从 issue #18619 与"Fix-regression"标题推断问题存在于引入相关改动的近期 main(由 diff 推断)。
- 回归面:按 diff,vLLM 侧仅动 vllm-flash-attn 的拉取版本,所有使用 FA3 后端的路径都在新 kernel 下运行,受益程度依模型/负载而异(Qwen VL 明显、Llama-3.1-8b 较小);低 QPS 场景持平无退化。依赖固定 commit 的升级会连带引入该区间内 flash-attention 的其他变更,正文未逐条列出;`test-qwen` 子模块疑似开发遗留产物随 PR 进入仓库。精度面由 lm_eval 对比与 CI 背书,数值输出无变化。

## 关联

- https://github.com/vllm-project/vllm/issues/18619 (问题来源与发现)
- https://github.com/vllm-project/flash-attention/pull/70 (实际修复所在,须先落地)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
