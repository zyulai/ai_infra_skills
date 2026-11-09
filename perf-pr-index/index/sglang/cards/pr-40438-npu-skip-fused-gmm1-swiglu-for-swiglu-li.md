# PR #40438: [NPU] Skip fused gmm1+swiglu for swiglu_limit (SiLU-with-clamp) checkpoints

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @iridiumine |
| 合并 | 2026-09-23 |
| 分类 | Attention / Kernel |
| Label | npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/40438 |

## 一句话结论

修正 NPU MoE 分派:fused gmm1 kernel 不含 clamp,swiglu_limit(DSV4)checkpoint 回退 unfused 路径恢复正确数值。

## 动机

前序 PR #39881 引入的 fused gmm1 kernel(`npu_grouped_matmul_swiglu_quant_v2`)只做 plain SiLU、没有 clamp,但 `_uses_fused_gmm1` 仅凭 kernel 类型与 `use_fused_gmm1` 选择它,从不看激活配置。激活为 SiLU-with-clamp(`swiglu_limit`,如 DSV4)的 checkpoint 因此丢失 clamp、数值错误,并静默绕过了 #38420 专门为该场景加的 `NPUSwigluMxfp8Quant` clamp 路径。这是一次性能融合引入的正确性回归。

## 优化点

单文件修改 `python/sglang/srt/layers/moe/moe_runner/ascend.py` 的分派逻辑:`_uses_fused_gmm1` 改为接收 runner config,当 `config.swiglu_limit` 设置且 >0 时返回 False——W4A8 MXFP4(如 DSV4 fp4 experts)回退到 unfused 路径(w13 gmm + `NPUSwigluMxfp8Quant(config.swiglu_limit)`),恢复 #39881 之前的 clamp 行为;MXFP8 + swiglu_limit 直接在 init 抛 `NotImplementedError`(该方法没有 unfused gmm1 路径,fail loudly 优于静默输出错误数值,当前无已知 checkpoint 命中该组合);plain-SiLU 模型(swiglu_limit 未设或 0)保留 fused kernel 不变。`__init__` 与 `run()` 两个调用点传入同一 config,保证激活选择与 fused/unfused 分支决策永不矛盾。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/ascend.py: _uses_fused_gmm1 增加 config 参数与 swiglu_limit 守卫
```

关键片段:

```python
has_clamp = config.swiglu_limit is not None and config.swiglu_limit > 0
if has_clamp:
    if isinstance(kernel, NPUMXFP8MoEMethod):
        raise NotImplementedError(...)  # 无 unfused 路径,不能服务 clamp checkpoint
    return False  # W4A8MXFP4: fall back to unfused to preserve the clamp
```

## 性能声明

PR 正文明确 "Speed Tests and Profiling: None"——本 PR 有意用回 #39881 的 fusion 收益换取 clamp checkpoint 的正确性。plain-SiLU 模型(如 Kimi-K3 MXFP4)不受影响,继续保留 fused 路径收益(PR 引述 #39881 的数字:每层 MoE 延迟降低约 15%)。为 clamp checkpoint 恢复 fusion 需要给 kernel 加 limit 参数,属 kernel 侧工作,不在本 PR 范围。

## 验证与回归风险

验证方式(PR 正文):本地用 mocked kernel 对全组合验证分派逻辑——W4A8 ± swiglu_limit、MXFP8 ± swiglu_limit、`use_fused_gmm1=False`、非 fused kernel 均按预期解析;DSV4 on NPU(fp4 experts → NPUW4A8MXFP4MoEMethod,swiglu_limit 取自 HF config)现在解析到 unfused clamp 路径,与 pre-#39881 serving 路径 bit-identical。无新增单测文件(checklist 中 "Add unit tests" 未勾选)。风险面:只影响 Ascend NPU 的 MXFP4/MXFP8 MoE 分派;若下游 checkpoint 误设 swiglu_limit 会从 fused 回退 unfused 造成性能下降(而非错误);MXFP8 + swiglu_limit 组合从静默错误变为启动报错。PR CI:Extra 失败(x),Base/AMD pending。

## 关联

前序:PR #39881(fused gmm1+swiglu 融合,本 PR 的回归来源)、PR #38420(NPUSwigluMxfp8Quant clamp 路径)。DSV4 相关:PR #32340([AMD][DSV4] shared experts fusion)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
