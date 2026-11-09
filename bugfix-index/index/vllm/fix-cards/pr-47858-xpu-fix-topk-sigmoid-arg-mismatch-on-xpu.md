# PR #47858: [XPU] Fix topk_sigmoid arg mismatch on XPU

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47858](https://github.com/vllm-project/vllm/pull/47858) |
| 作者 | zhenwei-intel |
| 合并日期 | 2026-07-07 |
| labels | intel-gpu, ready |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
#47408 给 _moe_C.topk_sigmoid 新增 routed_scaling_factor 而 XPU 内核未跟进,XPU 路径改为不传该参数的临时绕行。

## 问题现象
正文未描述具体运行时报错现象,仅说明:#47408 在 `_moe_C.topk_sigmoid` 中引入了新参数 `routed_scaling_factor`,而 XPU 内核(vllm-xpu-kernels)尚未更新以接受该参数,参数不匹配导致 XPU 平台调用该算子失败(标题与正文均以 arg mismatch 定性;Test Plan 与 Test Result 一节均为空,未附复现命令与输出)。

## 根因
正文披露:PR #47408 引入的新参数 `routed_scaling_factor` 是根因——CUDA 侧 `_moe_C.topk_sigmoid` 算子签名已更新,但 XPU 侧内核没有同步更新,Python 包装层把新参数传下去后与 XPU 内核的实际签名不匹配。

## 修复方案
在 `vllm/_custom_ops.py` 的 `topk_sigmoid` 包装函数开头加 XPU 分支作为 workaround:检测到 `current_platform.is_xpu()` 时,调用 `torch.ops._moe_C.topk_sigmoid` 时不传 `routed_scaling_factor`(仅传 topk_weights、topk_ids、token_expert_indices、gating_output、renormalize、e_score_correction_bias 六个原有参数)后直接 return;非 XPU 平台保持原调用(含 routed_scaling_factor)不变:
```python
if current_platform.is_xpu():
    # xpu doesn't support routed_scaling_factor currently, will revert
    # in next vllm-xpu-kernels bumpup
    torch.ops._moe_C.topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )
    return
```
注释注明该 workaround 将在下次 vllm-xpu-kernels 升级时还原(即届时 XPU 内核会支持新参数)。

## 影响与回归风险
- 影响窗口:自 #47408 合并引入新参数起(2026-07-07 修复),XPU 平台走 topk_sigmoid 的 MoE 模型
- 回归面:仅 XPU 路径的 topk_sigmoid 调用方式;XPU 上 `routed_scaling_factor` 被忽略(保持默认 1.0 语义),若模型依赖非 1.0 的路由缩放,输出可能与 CUDA 平台不一致——正文未讨论此点,注释承诺内核升级后还原;CUDA/CPU 路径不受影响。

## 关联
- [#47408](https://github.com/vllm-project/vllm/pull/47408)(引入新参数的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
