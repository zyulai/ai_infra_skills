# PR #27187: Revert "Fix TokenizerManager crash on top_logprobs with tensor values"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27187](https://github.com/sgl-project/sglang/pull/27187) |
| 作者 | ch-wan |
| 合并日期 | 2026-06-03 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论
回滚 #26825:恢复 top_logprobs 解码处的真值判断写法,并删除其新增的回归测试文件。

## 问题现象
正文未描述具体现象,仅说明回滚 #26825(Fix TokenizerManager crash on top_logprobs with tensor values)。PR 描述自带的 CI 状态槽显示两次失败的 run(Base: [Run #26915348882](https://github.com/sgl-project/sglang/actions/runs/26915348882) 标记 :no_entry_sign:,Extra: [Run #26915348809](https://github.com/sgl-project/sglang/actions/runs/26915348809) 标记 :x:),但正文没有写明失败原因或与回滚动机的关系。

## 根因
正文未披露回滚根因,仅有「Reverts sgl-project/sglang#26825」一句。

从 diff 看,回滚做了两件事:
- 把 `tokenizer_manager.py` 的 `detokenize_top_logprobs_tokens` 中对逐位置 logprob 值的判断从 `if token_logprobs_val[i] is not None:` 恢复为原来的 `if token_logprobs_val[i]:`;
- 删除 #26825 新增的回归测试文件 `test/registered/unit/managers/test_tokenizer_manager_top_logprobs_tensor.py`。

被删测试的文档字符串(属 diff 内容)记载了 #26825 的背景:当某个位置的 logprob 值是多元素 torch.Tensor 而非标注的 List[float] 时,旧真值判断会抛 `RuntimeError: Boolean value of Tensor with more than one value is ambiguous.`,异常经 print_exception_wrapper 捕获后 SIGKILL 整个 prefill 进程;#26825 改用 `is not None` 匹配实际哨兵值。本回滚恢复旧写法,意味着该判断回到真值语义(被跳过位置的 None 哨兵与张量真值行为回到 #26825 之前)(由 diff 推断)。

## 修复方案
- `python/sglang/srt/managers/tokenizer_manager.py` `detokenize_top_logprobs_tokens`:恢复原判断:

```python
for i in range(len(token_logprobs_val)):
    if token_logprobs_val[i]:
        ret.append(
            self.detokenize_logprob_tokens(
                token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
            )
        )
```

- 删除 `test/registered/unit/managers/test_tokenizer_manager_top_logprobs_tensor.py`(82 行回归测试整体移除)。

## 影响与回归风险
- 影响窗口:正文未说明(#26825 合入至本回滚合并之间)。
- 回归面:影响 TokenizerManager 的 top_logprobs 解码路径;#26825 所修的「多元素张量值导致进程崩溃」问题随回滚恢复存在,依赖 `is not None` 行为的新路径需回到旧真值语义。被删回归测试不再提供覆盖。

## 关联
- [#26825](https://github.com/sgl-project/sglang/pull/26825)(被回滚的修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
