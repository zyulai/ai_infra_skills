# PR #39844: [XPU] fix all_reduce all-zero accuracy issue under torch.compile

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39844](https://github.com/vllm-project/vllm/pull/39844) |
| 作者 | chaojun-zhang |
| 合并日期 | 2026-04-18 |
| labels | intel-gpu, ready |
| 热度 | 👍 2 · 💬 1 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
XPU 在 torch.compile 下 all_reduce 输出全零(GSM8K 得 0 分);编译时 clone 输入再原地 all_reduce 修复。

## 问题现象
XPU 上 compile 模式下 all_reduce 返回全零。lm-eval GSM8K(meta-llama/Llama-3.1-8B, bf16, 5-shot):main + compile 的 exact_match 为 **0.0**,同配置 eager 为 0.52——compile 模式完全不可用;修复后 compile 恢复到 0.52。

## 根因
正文披露:`dist.all_reduce` 是原地(in-place)操作。被 inductor 追踪时,由于编译器看不到"产出了新张量",原始输入张量可能被优化掉,导致输出全零。

## 修复方案
`vllm/distributed/device_communicators/xpu_communicator.py` 的 `all_reduce`:编译期先 clone 出新张量再做原地 all_reduce,让 inductor 能看到新产出的张量;非编译路径不变:

```python
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    output = input_.clone() if torch.compiler.is_compiling() else input_
    dist.all_reduce(output, group=self.device_group)
    return output
```

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:仅 `xpu_communicator.all_reduce`;编译路径多一次 `input_.clone()` 的显存分配与拷贝。性能数据(Llama-2-13b-chat-hf, tp=2):compile 吞吐 363.56→363.12 tok/s(-0.12%)、Median TTFT +1.76%;eager 吞吐 -0.11%、TTFT +0.38%;eager latency 2.6467s→2.6783s。正文 checklist 未勾选,未列专门单测。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
