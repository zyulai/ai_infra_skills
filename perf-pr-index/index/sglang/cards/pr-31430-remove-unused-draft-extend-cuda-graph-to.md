# PR #31430: Remove unused draft-extend CUDA graph top-k

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-07-31 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | performance, deepseek, speculative-decoding, run-ci |
| Diff | diffstat 缺失(素材未提供增删行统计,diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/31430 |

## 一句话结论

从 draft-extend CUDA graph capture 中删除从不被消费的 top-k 计算,减少 graph 内死代码,准确性不变。

## 动机

draft-extend CUDA graph 在 capture 时对每一行 draft-extend 输出都计算 top-k,但 replay 只返回 logits 与 hidden states:在用 `select_index` 选出相关行之后,`EagleDraftWorker` 本身就会计算所需的 top-k(见 `eagle_worker_v2.py` 中引用的代码段),因此 graph 里捕获的 top-k 结果从未被消费,属于纯浪费的图内计算。

## 优化点

改动是纯粹的死代码清除,不改任何机制:在 `EAGLEDraftExtendCudaGraphRunner.run_once()`(capture 体)中删除 top-k 分支——即 `topk == 1` 且非 HIP 时的 `torch.argmax` 快路径,以及一般的 `softmax + fast_topk` 路径;同时移除 runner 上不再需要的 `self.topk` 状态、`fast_topk`/`is_hip` 等导入。worker 侧的 selected-row top-k 计算保持不变。测试侧同步清理:删除对 `topk_p`/`topk_index` 的 assert 及 eager 参考中镜像生产快路径的 top-k 逻辑,注释更新为"graph 不计算 topk、replay 不返回 topk 字段"。

## 关键改动

```text
python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py: run_once 删除 capture 内 top-k/argmax 分支,移除 self.topk 与相关导入
python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_extend_runner.py: 删除 topk 断言与 eager 参考 top-k 镜像,统一走 v2 比较函数
```

被删除的 capture 内计算(原样摘录,现由 worker 在 replay 后对选中行计算):

```python
if self.topk == 1 and not _is_hip:
    ret.topk_index = torch.argmax(
        ret.next_token_logits, dim=-1, keepdim=True
    )
    ret.topk_p = torch.ones_like(ret.topk_index, dtype=torch.float32)
else:
    probs = torch.softmax(ret.next_token_logits, dim=-1)
    ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)
```

## 性能声明

PR 正文未提供量化性能数字(无吞吐/时延 benchmark);收益定性为消除 draft-extend CUDA graph capture 中从未消费的全词表 top-k 计算。验证以准确性为主:GSM8K 100 样本,baseline 与本 PR 均 strict 99/100、flexible 99/100,两次运行漏掉同一样本且产生相同答案;`pre-commit run --all-files` 通过。PR 未注明 benchmark 的模型/硬件/workload 条件。

## 验证与回归风险

验证依赖:GSM8K 100 样本配对(见上)+ attention unittest kit 中 speculative draft-extend runner 模式的输出比较(改为只比较 graph 实际锚定的 `next_token_logits`/`hidden_states`)。风险路径:EAGLE/MTP 类 speculative decoding 的 draft-extend 阶段,尤其依赖 graph 输出 `topk_p`/`topk_index` 字段的外部调用方——本 PR 明确这两个字段由 `EagleDraftWorker` 在选择行之后计算,不在 graph 内;ROCm(HIP)上原 argmax 快路径本就被排除,行为不受影响。删除的 `#26397` 快路径注释表明该分支历史上为 CUDA graph replay 引入,回归需关注 topk==1 场景的 draft 选择等价性。

## 关联

前序:#26397(graph replay 的 topk==1 快路径来源)。同区域:draft-extend CUDA graph runner 与 `EagleDraftWorker` top-k 计算相关 PR;标签同含 deepseek 与 speculative-decoding。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
