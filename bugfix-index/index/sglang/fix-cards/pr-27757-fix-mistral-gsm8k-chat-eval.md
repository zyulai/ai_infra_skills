# PR #27757: Fix Mistral GSM8K chat eval

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27757](https://github.com/sgl-project/sglang/pull/27757) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-07-10 |
| labels | high priority, dependencies, run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
旧 5-shot chat 模板打分器在 Mistral 上失真,加之 antlr4 装包被污染;切换 sgl-eval 并强制重装 antlr4。

## 问题现象
nightly-test-text-accuracy-2-gpu-h100 在 Mistral 系模型上变红:mistralai/Mistral-7B-Instruct-v0.3 得分 0.414,低于 0.47 阈值。devbox 复现显示模型生成正常、finish_reason=stop,并非服务端回归。此外 CI 上 sgl-eval 导入抛 `ImportError: Unsupported ANTLR version 4.7.2`,先前的 rerun 全部模型失败。

## 根因
正文披露两点:
1. 旧打分器把 completion 风格的 5-shot `Question:/Answer:` prompt 包进 Mistral 的 `[INST]` chat 模板,破坏了 Mistral 系模型的 few-shot 上下文;且响应延续到下一个 Q/A 块时,「取最后一个数字」的抽取逻辑偶尔抽错数。
2. CI 装包顺序 bug:sgl-eval 的 latex2sympy2_extended 需要 antlr4 4.9.3/4.11/4.13.2,遇 4.7.x 抛 ImportError;pyproject 的 4.9.3 pin 不够——install_extra_deps 装 lmms-eval v0.5 时,其传递依赖 latex2sympy2 1.9.1 硬 pin antlr4==4.7.2,把 4.9.3 覆盖且后续步骤不恢复。

## 修复方案
1. nightly GSM8K eval 切到 sgl-eval(`api="sgl_eval"`,零样本 chat + `\boxed{}` + math_verify 判分):run_eval.py 新增 `_run_sgl_eval` 子进程封装,解析 metrics.json 的 aggregate.score;其余 gsm8k 调用方保持 5-shot 打分器(GSM8KEval 移入 simple_eval_mixed_prefix_gsm8k.py 保留)。
2. 为全部 15 个 nightly 模型按 H100 2-GPU 全量 1319 题实测分 − 0.05 填真实阈值。
3. ci_install_dependency.sh 在装完 lmms-eval 后强制 `pip install "antlr4-python3-runtime==4.9.3" --force-reinstall --no-deps`。
4. pyproject [test] 增加 sgl-eval git SHA pin 与 antlr4 pin。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:nightly GSM8K 评测路径与 15 个模型的阈值、run_eval 的 gsm8k 分支(sgl-eval 仅 opt-in,其他调用方不变)、ChatCompletionSampler 新增 stop 参数、CI 依赖安装脚本;sgl-eval pin 升级会改变判分,需先重新 baseline 阈值。

## 关联
- [run 28645576771](https://github.com/sgl-project/sglang/actions/runs/28645576771)(15 模型全过)
- [run 28642161462](https://github.com/sgl-project/sglang/actions/runs/28642161462)(antlr4 覆盖导致失败的先前 rerun)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
