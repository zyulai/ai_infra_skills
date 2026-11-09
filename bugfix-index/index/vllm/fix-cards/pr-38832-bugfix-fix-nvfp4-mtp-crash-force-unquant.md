# PR #38832: [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38832](https://github.com/vllm-project/vllm/pull/38832) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-04-03 |
| labels | bug, qwen, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
NVFP4 checkpoint 漏排除 mtp.fc 导致权重形状不匹配断言失败,加载 MTP 时对该层强制使用非量化配置修复。

## 问题现象
加载 `nvidia/Qwen3.5-397B-A17B-NVFP4` 并配置 `method="mtp"` 时抛出 `AssertionError`:在 2x B200、TP=2、`--language-model-only`、`--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`、`--max-model-len 1024` 的场景下,Before 表现为 `parameter.py:153` 处 `AssertionError`(MTP 权重加载阶段);修复后服务正常启动并可推理。

## 根因
正文披露:该 NVFP4 checkpoint 将整个 MTP 分支以 BF16 存储,但 `hf_quant_config.json` 的 exclude_modules 只排除了 `mtp.layers.0*`,漏掉 `mtp.fc`。于是 `mtp.fc` 的 `ColumnParallelLinear` 被按 NVFP4 量化创建(packed uint8、输入维度减半),权重加载时与 BF16 checkpoint 的权重形状不匹配而崩溃。

## 修复方案
修改 `vllm/model_executor/models/qwen3_5_mtp.py`:当量化配置为 `modelopt_fp4` 时,对 `mtp.fc` 强制使用 `quant_config=None`(非量化)创建 `ColumnParallelLinear`,其余情况仍用原 quant_config。这是临时 workaround,待 NVIDIA/Model-Optimizer#1124 合并、checkpoint 以修正后的 exclude_modules 重新导出后可移除。

关键 diff 摘录:
```python
fc_quant = (
    None
    if (quant_config and quant_config.get_name() == "modelopt_fp4")
    else quant_config
)
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    ...
    quant_config=fc_quant,
)
```

## 影响与回归风险
- 影响窗口:使用缺少 mtp.fc 排除项的 NVFP4 checkpoint 的场景(正文未给出具体版本范围)。
- 回归面:仅改动 `qwen3_5_mtp.py` 中 `mtp.fc` 的量化配置选择;modelopt_fp4 量化下该层固定非量化,其他量化方式与其他层不受影响。

## 关联
- [PR #38650](https://github.com/vllm-project/vllm/pull/38650)
- [NVIDIA/Model-Optimizer#1124](https://github.com/NVIDIA/Model-Optimizer/pull/1124)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
