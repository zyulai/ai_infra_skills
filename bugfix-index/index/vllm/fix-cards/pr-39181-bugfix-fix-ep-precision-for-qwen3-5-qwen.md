# PR #39181: [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39181](https://github.com/vllm-project/vllm/pull/39181) |
| 作者 | USTCKAY |
| 合并日期 | 2026-04-08 |
| labels | bug, qwen, ready |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

开启 sequence parallel 时 Qwen3.5/Qwen3-Next 的共享专家权重被 TP 切分且不做 all-reduce,导致 EP 下精度严重劣化(GSM8K 0.318→0.859);改为 SP 下禁用共享专家 TP 切分。

## 问题现象

在 8 卡 A800 上以 `-tp 4 -dp 2 --enable-expert-parallel --all2all-backend naive`(vllm 0.18.0,bfloat16)部署 Qwen3.5-35B-A3B 时模型输出异常:问"0.9 和 0.11 哪个大"时输出退化为大量重复的 `*` 字符并跑满 800 token;GSM8K 评测(1319 题,5-shot)准确率仅 0.318,无效响应率 0.015。

## 根因

正文披露:当 `sequence_parallel` 开启时,共享专家(shared experts)会应用 tensor parallel 对权重做切分,而计算结束时不做 all-reduce,从而产生精度问题。即共享专家被切分到各 rank 只算部分结果,又缺少规约步骤,输出不完整。

## 修复方案

修改 `vllm/model_executor/models/qwen2_moe.py` 的共享专家实现(Qwen3-Next 复用该类):新增 `is_sequence_parallel: bool = False` 参数,在 `gate_up_proj`(ColumnParallelLinear)与 `down_proj`(RowParallelLinear)上设置 `disable_tp=is_sequence_parallel`,即 SP 模式下不再切分共享专家权重;`vllm/model_executor/models/qwen3_next.py` 构造 shared expert 时传入 `is_sequence_parallel=self.is_sequence_parallel`。关键 diff:

```diff
+        is_sequence_parallel: bool = False,
...
             quant_config=quant_config,
+            disable_tp=is_sequence_parallel,
             prefix=f"{prefix}.gate_up_proj",
```

修复后同一 curl 测试输出正常推理过程,`finish_reason` 从 `length` 变为 `stop`;GSM8K 准确率 0.318 → 0.859,无效响应 0.015 → 0.000。

## 影响与回归风险

- 影响窗口:正文提到测试于 vllm 0.18.0,且"与最新 main 分支在共享专家部分无差异",未给出更早的引入版本。
- 回归面:仅影响 Qwen2-Moe/Qwen3-Next 系列在开启 sequence parallel + EP 时的共享专家权重分片方式;SP 下共享专家改为每个 rank 持有完整权重,显存占用会增加(不再按 TP 切分),这是行为上的权衡点。非 SP 或非 EP 场景不受影响。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
