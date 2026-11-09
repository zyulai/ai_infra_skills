# PR #19961: fix: change qwen 3.5 linear attention a_log to fp32

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19961](https://github.com/sgl-project/sglang/pull/19961) |
| 作者 | shiyu7 |
| 合并日期 | 2026-03-18 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
Qwen 3.5 线性注意力的 a_log 参数未按原设计使用 fp32,改为显式以 float32 创建该参数。

## 问题现象
正文未描述具体现象,仅在 Motivation 中说明:Qwen 3.5 线性注意力的 `a_log` 数据类型需要改为 fp32,以符合 Qwen 3.5 的原始设计。正文没有给出触发场景、日志或复现步骤;结合该卡被归为 hang 类,推测与精度相关的异常在特定条件下才显现,但正文未展开。

## 根因
正文未披露根因分析,只说明了修改动机:当前实现中 `a_log` 的数据类型与 Qwen 3.5 原始设计不一致(原设计要求 fp32)。

从 diff 可以描述改动所暗示的机制:Qwen3.5 模型中线性注意力分支的 `A_log` 参数此前用 `torch.empty(...)` 创建,未显式指定 dtype,因而跟随模型默认的低精度 dtype;低精度下该参数参与线性注意力衰减系数的计算时可能引入数值误差。(由 diff 推断)

## 修复方案
改动集中在 `python/sglang/srt/models/qwen3_5.py` 的线性注意力模块 `__init__` 中:创建 `self.A_log` 参数时显式加上 `dtype=torch.float32`,其余逻辑不变。

```diff
         self.A_log = nn.Parameter(
-            torch.empty(self.num_v_heads // self.attn_tp_size),
+            torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
         )
```

`A_log` 仍通过 `sharded_weight_loader(0)` 加载分片权重,shape 与分片方式未变,仅存储/计算 dtype 变为 fp32。

## 影响与回归风险
- 回归面:仅影响 Qwen 3.5(qwen3_5.py)线性注意力分支的 `A_log` 参数精度;参数显存占用会略有增加(该参数从默认低精度变为 fp32),权重加载路径 `sharded_weight_loader` 需要兼容 fp32 目标 dtype,属于常规 cast 加载路径。

## 关联
正文未列出
