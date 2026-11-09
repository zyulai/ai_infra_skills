# PR #23776: [DeepSeek V4] Fix meaningless numbers in chat output by adding swiglu_limit clamp to DeepseekV2MLP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23776](https://github.com/sgl-project/sglang/pull/23776) |
| 作者 | GaoYusong |
| 合并日期 | 2026-04-27 |
| labels | high priority, deepseek |
| 热度 | 👍 35 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
DeepSeek-V4 共享专家/稠密 MLP 漏传 `swiglu_limit`,SiLU 输出无界膨胀污染残差流,聊天输出出现确定性乱数 token;补上 clamp 修复。

## 问题现象
DeepSeek-V4 聊天输出在句子边界位置出现确定性乱数 token:纯文本 `16`、`14:05`、`<｜end▁of▁file｜>` 后跟训练数据文件路径并进入解码死循环。#23752 的示例输出:"...Thiopentone is a barbiturate with anticonvulsant properties,**16** used sometimes in status epilepticus..."。`16` 的 token ID 为 `[49, 54]`(ASCII `'1' '6'`),是 `temperature=0` 下 argmax 选出的纯文本。触发条件:使用带 `swiglu_limit=10.0` 的 DeepSeek-V4 checkpoint(如 V4-Pro / V4-Flash 的 `2604B` 子模式)。

## 根因
PR 正文披露:routed-expert 路径已读取 `config.swiglu_limit` 并经 `FusedMoE` → `MoeRunnerConfig.gemm1_clamp_limit` → triton/deepgemm kernel 做 SiLU clamp;但共享专家路径和稠密 MLP 路径用的是 `DeepseekV2MLP`,从未接收 `swiglu_limit`,`DeepseekV2MLP.forward` 直接对 `gate_up` 调用裸 `SiluAndMul()`,SiLU 输出无界。对训练时用 `swiglu_limit=10.0` 的 V4 checkpoint,推理时共享专家 SiLU 输出膨胀到 **±2000+**,经 `final_hidden_states += shared_output` 污染残差流、穿过 `mhc_post`,在句子边界位置劣化 `lm_head` logits,产生乱数 token。`mhc_post` 调试钩子显示 `hc_post` 输出 min/max 达 ±1368~±2576;而 `mhc_post` 自身公式 `post * x + sum(comb * residual, dim=1)` 中 `post ∈ [0, 2]`、`comb` 经 Sinkhorn 归一化到 `[0,1]`,不可能自行放大,故毒源只能是被未 clamp 的共享专家 SiLU 输出污染的 residual/x 输入。`deepseek_v4.py:1432-1433` 的注释 "2604B checkpoint requires different clamping for shared and routed experts" 早已标记此问题:禁用共享专家融合已实现,但未融合的 `DeepseekV2MLP` 路径缺少对应 clamp。

## 修复方案
对 `python/sglang/srt/models/deepseek_v2.py` 做三处最小改动:
1. `DeepseekV2MLP.__init__` 新增 `swiglu_limit: Optional[float] = None` 参数并保存。
2. `DeepseekV2MLP.forward` 在 `SiluAndMul` 前对 `gate_up` 分块 clamp,语义对齐 routed-expert kernel 的 `_swiglu_silu_clamp_mul`:
   ```python
   if self.swiglu_limit is not None:
       _g, _u = gate_up.chunk(2, dim=-1)
       _lim = float(self.swiglu_limit)
       gate_up = torch.cat(
           [_g.clamp(max=_lim), _u.clamp(min=-_lim, max=_lim)], dim=-1
       )
   ```
3. 两个调用点(`shared_experts` L530、稠密 MLP `self.mlp` L2589)传入 `swiglu_limit=getattr(config, "swiglu_limit", None)`,与 L485 routed experts 的既有写法一致。

验证:#23752 的 Anaesthesia MCQ 复现用例中 3 处乱数 `16` 全部消除;多 prompt 冒烟(医疗/对话/长文/代码/数学)0 泄漏(补丁前长文用例出现多次 `16` 插入及 `the16 the16 ...` 解码循环)。测试基于 `deepseek_v4` 分支的 `97d1a672`。

## 影响与回归风险
- 回归面:行为完全由 `config.swiglu_limit` 门控,未引入环境变量。不设置该配置的模型(如普通 V2/V3)行为不变;设置了该配置的模型,共享专家与稠密 MLP 路径的 SiLU 输出增加上限 clamp,与 routed-expert kernel 语义对齐。`swiglu_limit=None` 是既有的隐式行为,新增带默认值的 kwarg 非破坏性。若下游自定义模型子类直接实例化 `DeepseekV2MLP`,需注意新 kwarg 可选、不影响签名兼容。

## 关联
- Fixes #23752

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
