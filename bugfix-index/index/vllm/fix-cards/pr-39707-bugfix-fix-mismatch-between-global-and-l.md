# PR #39707: [Bugfix] Fix mismatch between global and local attention heads in tensor-parallel mode for param2moe model

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39707](https://github.com/vllm-project/vllm/pull/39707) |
| 作者 | bhargav-patel-29 |
| 合并日期 | 2026-04-14 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
param2moe 的 `Attention` 用全局头数而 QKV 已按 TP 分片,TP>1 时注意力算错;改用本地头数并保证 Q/K/V 连续。

## 问题现象
TP(tensor parallelism)>1 时 `Param2moeAttention` 的注意力计算不正确。验证方式为 `vllm serve bharatgenai/Param2-17B-A2.4B-Thinking --trust-remote-code --tensor-parallel-size {1,2,4,8}`,对比 TP=1 与 TP>1 的输出:修复前 TP>1 输出与 TP=1 不一致、注意力行为错误;修复后 TP>1 无报错、输出与 TP=1 一致(在预期方差内)。正文未描述具体报错日志。

## 根因
正文披露:`Attention` 模块使用的是全局(global)头数(`num_heads` / `num_kv_heads`),而 QKV 张量此时已经按 TP 分片(sharded),两者不一致造成 attention 计算失配,进而产生错误行为。

## 修复方案
改 `vllm/model_executor/models/param2moe.py`:

1. 构造 `Attention` 时改用本地(local)头数:

```python
self.attn = Attention(
    num_heads=self.num_local_heads,
    head_size=self.head_dim,
    scale=self.scaling,
    num_kv_heads=self.num_local_kv_heads,
    ...
)
```

2. `forward` 中把 split 出来的 Q/K/V 转为连续内存:

```python
q = q.contiguous()
k = k.contiguous()
v = v.contiguous()
```

此外删除了文件头重复的一行 license 注释和 forward 内若干步骤注释(不影响逻辑)。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:仅改动 `vllm/model_executor/models/param2moe.py`,影响该模型在所有 TP 设置下的注意力路径:TP=1 时 `num_local_heads` 与全局头数相同,行为不变;TP>1 时头数与分片后的 QKV 尺寸匹配。新增的 `.contiguous()` 调用可能带来少量内存拷贝开销。顺带清理了注释与重复 license 行。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
