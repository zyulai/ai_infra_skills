# PR #27883: Fix fp16 NaN flake in spec CI: bf16 eagle fixture; sanitize NaN logits in sampler

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27883](https://github.com/sgl-project/sglang/pull/27883) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-06-11 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
EAGLE 投机解码中 fp16 激活溢出产生 NaN 触发 CI 假性 device assert;fixture 改 bf16 并在采样入口清洗 NaN。

## 问题现象
`test_spec_eagle_fa3.py` 出现偶发(flaky)的 NaN device assert,例如 CI run https://github.com/sgl-project/sglang/actions/runs/27318076885/job/80706381850 。触发条件:EAGLE verify 阶段出现退化的 draft 分支(degenerate draft branch)时,fp16 激活溢出,经 RMSNorm 后 Inf 变为 NaN。除 CI 假失败外,NaN 若进入采样 kernel 属未定义行为,可能表现为返回 out-of-vocab token id。

## 根因
正文披露:fp16 激活在 EAGLE verify 的退化 draft 分支上溢出,经 RMSNorm 变成 Inf 再变 NaN。另一层问题是采样 kernel 对 NaN 输入是未定义行为,NaN logits 一旦到达采样 kernel 可能回传 out-of-vocab token id。修复同时治标(采样入口清洗 NaN)与治本(测试 fixture 弃用 fp16 改 bf16,避免激活溢出)。

## 修复方案
1. `python/sglang/test/server_fixtures/spec_eagle_fixture.py`:`SpecEagleServerBase.dtype` 由 `"float16"` 改为 `"bfloat16"`,注释说明 fp16 激活会在退化 draft 分支溢出(→ Inf → NaN)触发 CI NaN assert。
2. `python/sglang/srt/utils/async_probe.py`:新增 `sanitize_nan_logits()` 与 `_AsyncNanWarner`(免同步的节流式 NaN 告警):先 `maybe_detect_nan`(CI 硬 assert),再 `torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)`;用 ±1e30 而非 dtype 极值,因为调用方会把 logits 除以 temperature,dtype 极值会溢出为 ±Inf 再被 softmax 变回 NaN。
3. `python/sglang/srt/layers/sampler.py`:`_preprocess_logits` 中调用 `sanitize_nan_logits(logits, "sampler: next_token_logits")`。
4. `python/sglang/srt/speculative/eagle_info_v2.py`:`sample()` 中对 verify 阶段的 `next_token_logits` 调用 `sanitize_nan_logits(..., "verify: target model logits")`。
5. `python/sglang/srt/environ.py`:新增开关 `SGLANG_SANITIZE_NAN_LOGITS = EnvBool(True)`(默认开启)。

```diff
+    maybe_detect_nan(logits, msg)
+    if not envs.SGLANG_SANITIZE_NAN_LOGITS.get():
+        return
+    maybe_warn_nan(logits, msg)
+    torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:所有走 Sampler `_preprocess_logits` 的路径与 EAGLE verify 采样路径都会多一次 NaN 清洗与异步检测(默认开启,可用 `SGLANG_SANITIZE_NAN_LOGITS=0` 关闭);±1e30 替换会改变原本就含 NaN/Inf 的输出行为(从未定义变为有界);spec EAGLE 测试 fixture 的 dtype 变化会改变这些测试的数值基线;`_AsyncNanWarner` 的 pinned host 内存读取逻辑只告警一次。

## 关联
- CI 失败示例:https://github.com/sgl-project/sglang/actions/runs/27318076885/job/80706381850
- 其余 issue/PR:正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
