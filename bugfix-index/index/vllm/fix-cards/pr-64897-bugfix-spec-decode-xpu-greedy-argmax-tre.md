# PR #64897: [Bugfix][Spec Decode][XPU] greedy_argmax: treat NaN as -inf on the torch fallback

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64897](https://github.com/vllm-project/vllm/pull/64897) |
| 作者 | taneem-ibrahim |
| 合并日期 | 2026-11-09 |
| labels | bug, intel-gpu, mrv2, ready |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
greedy_argmax 的 torch 回退遇 NaN 会返回 NaN 索引,违背「NaN 视为 -inf」契约、阻塞 XPU CI;改为 masked_fill(-inf) 后再 argmax。

## 问题现象
正文开头即说「XPU on main is broken」。复现方式:给既有测试 `test_greedy_argmax_matches_torch` 的 vocab 参数加入 `1000`,在同一命令下对比:

```bash
python -m pytest -v tests/v1/worker/test_gpu_gumbel_sample.py -k test_greedy_argmax_matches_torch
```

H100 80GB、PyTorch 2.14.0+cu130、Triton 3.8.0 上,main 在两个新增 fallback 用例的 NaN 行失败(`2 failed, 6 passed, 22 deselected`),6 个 Triton 用例全部通过;修复后分支为 `8 passed`,整个文件 `30 passed`。触发条件:走 torch fallback 的场景,即 XPU、fp16、小词表(<4096)、CPU 或非连续输入,且 logits 行中含 NaN。

## 根因
正文明确披露:该问题由 #57770 引入。torch fallback 直接用 `logits.argmax(dim=-1)`,而 `torch.argmax` 遇 NaN 返回 NaN 所在索引;但函数 docstring 早已承诺「NaN logits 视为 -inf」,Triton 路径也正是这样实现的。两条路径对同一 logits 给出不同 token,作者在 Reviewer Notes 中说明:如果改测试迁就 fallback,会让 draft token 依赖平台与 dtype,因此必须修实现。

## 修复方案
改 `vllm/v1/worker/gpu/sample/greedy_argmax.py` 的 torch fallback 分支,并把 docstring 改为「所有路径都把 NaN 当 -inf」:

```python
-        return logits.argmax(dim=-1)
+        return logits.masked_fill(logits.isnan(), float("-inf")).argmax(dim=-1)
```

`masked_fill(isnan, -inf)` 保留无穷大与「同分取首个索引」语义,且不修改输入本身。mask 与 logits 拷贝只发生在 fallback 路径;Triton 内核及其 gating 条件完全不变。

## 影响与回归风险
- 影响窗口:自 #57770 引入该 fallback 行为起,至本 PR 合并止;具体版本号正文未给出。
- 回归面:按 diff,仅 torch fallback 分支受影响——覆盖 XPU、fp16、词表小于 4096、CPU 及非连续输入。bf16/fp32 模型在 CUDA/ROCm 上仍走 Triton 路径,典型部署无行为差异。代价是 fallback 路径多一份 logits 拷贝;作者评估其开销远小于 draft 步已读取的权重。曾考虑为 XPU 打开 Triton 内核以免拷贝,但需要 XPU 性能数据与 XPU owner 签核,留待后续 PR。

## 关联
- [#57770](https://github.com/vllm-project/vllm/pull/57770)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
