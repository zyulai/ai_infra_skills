# PR #34668: fix(test): stabilize nightly precision regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34668](https://github.com/sgl-project/sglang/pull/34668) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-08-26 |
| labels | documentation, run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 39 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

nightly 精度回归测试在 GLM-5.2-FP8 TP=8 上因 stale baseline 与比较 rank-local BF16 缓冲产生约 8x 误报;改为始终恢复远端 baseline、只比较融合后的 attention 输入,并新增受限的基线刷新通道。

## 问题现象

`test_nightly_precision_regression` 对 `zai-org/GLM-5.2-FP8`、TP=8 失败,由两个相互独立的 harness 原因叠加(与 serving runtime 无关):

1. 当 self-hosted runner 在 `/tmp/sglang_precision_baselines` 下已有 `.pt` 文件时,测试跳过远端 baseline 恢复。而这份 stale baseline 是在 FlashInfer fusion 初始化失败后捕获的 replicated layer-entry tensors,target 却是 TP-partial tensors;两者都被当作 partial 处理,产生约 8x(tp_size 倍)的不匹配。fusion 初始化失败是历史问题,由 FlashInfer #3676 与 SGLang #30875 跟踪,并已被 SGLang #30580 缓解。
2. 即使做了干净的 fused refresh,comparator 仍观察到 fused all-reduce 与 residual RMSNorm 之前的 rank-local BF16 buffers;对这些缓冲做 CPU 侧 reduction 得到的不是 kernel 的语义输出,且跨 runner 漂移。

正文说明当前 H200 运行在全部八个 rank 上都能初始化 TRTLLM fusion。

## 根因

正文披露:根因在测试 harness 的比较契约,而非运行时代码——比较对象选错了(比较 TP-partial 的 layer-entry 张量,需要 CPU reduction,既非语义输出又跨机漂移),且 baseline 恢复逻辑有洞(本地残留 `.pt` 会短路远端恢复,使 stale 与 fresh baseline 语义不一致仍参与比较)。该 PR 明确「Serving runtime and all-reduce implementation code are unchanged」。

## 修复方案

- 恢复逻辑:始终恢复 signature 兼容的远端 baseline,经临时目录 staging 后原子替换本地 tensors(失败时清理残留目录,避免 stale 文件存活);`capture_signature` 纳入 capture filters、TP size、comparator filter 与 fusion backend,契约变化即重建基线。
- 比较对象:只比较 replicated 的 post-fusion attention 输入 `self_attn.inputs.hidden_states`(fused all-reduce 与 residual RMSNorm 之后、attention 实际消费的值),comparator 以 `--filter` + `tp:replicated` 声明;layer-entry `inputs.1` 仍捕获但排除出比较,仅由新增 `_assert_fused_tp_layout` 用作 guard——若它们变为 replicated 说明 fusion 静默回退,测试失败。
- CI 权限:普通 `/rerun-test` 改为 read-only(不写共享基线);新增 maintainer-only、仅限 in-repo PR 的刷新路径 `/rerun-test --refresh-precision-baseline test/registered/debug_utils/test_nightly_precision_regression.py`,在 workflow 与 slash_command_handler 双重校验(mode、runner、无 head sha、精确测试名、write 权限)。

改动文件:`test/registered/debug_utils/test_nightly_precision_regression.py`、`python/sglang/test/precision_baseline_store.py`、`scripts/ci/utils/slash_command_handler.py`、`.github/workflows/rerun-test.yml`、文档与单测。

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围,不写。
- 回归面:仅测试/CI 基线设施——nightly 精度测试的比较对象与恢复逻辑、HF baseline store 的读写行为、`/rerun-test` 权限校验。serving runtime 与 all-reduce 实现零改动;signature 变化会使既有远端基线失配而触发重建。fork PR 无法触发基线写 token。

## 关联

- [FlashInfer #3676](https://github.com/flashinfer-ai/flashinfer/issues/3676)
- [SGLang #30875](https://github.com/sgl-project/sglang/issues/30875)
- [SGLang #30580](https://github.com/sgl-project/sglang/pull/30580)
- #23402(diff/文档中提及 SM90 auto-enable 被移除)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
