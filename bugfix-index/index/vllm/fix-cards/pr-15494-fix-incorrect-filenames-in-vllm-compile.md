# PR #15494: Fix incorrect filenames in vllm_compile_cache.py

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15494](https://github.com/vllm-project/vllm/pull/15494) |
| 作者 | zou3519 |
| 合并日期 | 2025-03-27 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
`hijacked_compile_fx_inner` 把被 hook 包裹过的 callable 的 co_filename 当作 Inductor 编译产物文件路径,记录到 compile cache 的文件名不正确;改为沿 `__closure__` 找到真正来自 Inductor cache 目录的函数。

## 问题现象
`vllm/compilation/compiler_interface.py` 记录的编译产物文件路径不正确。正文只有 Test Plan,未描述具体故障(如缓存失效/命中异常的行为表现),仅给出修复后本地运行 `vllm serve meta-llama/Llama-3.2-1B-Instruct` 验证 `vllm_compile_cache.py` 的文件名恢复为 Inductor cache 路径的形态:
```
/home/rzou/.cache/vllm/torch_compile_cache/a785fdbf18/rank_0_0/inductor_cache/w2/cw22szhsitonk3z4ekx476g3hzaet3vy5qmsoewpx3kgbokhcob2.py
```

## 根因
正文未披露根因。从 diff 描述机制(由 diff 推断):`hijacked_compile_fx_inner` 原来直接用 `inductor_compiled_graph.current_callable.__code__.co_filename` 作为 file_path。但当该 callable 被 Inductor 的 `align_inputs_from_check_idxs`(torch/_inductor/utils.py)等外层函数 hook 包裹时,`__code__.co_filename` 指向的是 PyTorch 包装函数所在文件,而不是 Inductor 编译产物在 vLLM cache 目录下的真实 .py 文件——因此记录的文件名不正确。新代码注释明确写着"hooked in the align_inputs_from_check_idxs function in torch/_inductor/utils.py"。

## 修复方案
改 `vllm/compilation/compiler_interface.py` 的 `hijacked_compile_fx_inner`:
```python
compiled_fn = inductor_compiled_graph.current_callable
file_path = compiled_fn.__code__.co_filename
if not file_path.startswith(self.cache_dir):
    for cell in compiled_fn.__closure__:
        if not callable(cell.cell_contents):
            continue
        code = cell.cell_contents.__code__
        if code.co_filename.startswith(self.cache_dir):
            file_path = code.co_filename
            break
```
即当外层 callable 的 co_filename 不在 vLLM cache 目录下时,遍历其 `__closure__` 闭包单元,找到 cell 中可调用对象的 code 对象且 `co_filename` 以 `self.cache_dir` 开头者,认为这才是 Inductor 真正编译出的文件路径并采用之。`hash_str = inductor_compiled_graph._fx_graph_cache_key` 的取值保持不变。

## 影响与回归风险
- 影响窗口:正文未说明(涉及该 hijack 存在的版本;正文未给出受影响版本范围)。
- 回归面:仅 `compiler_interface.py` 中 Inductor 编译产物 file_path 的推断逻辑,即 `vllm_compile_cache.py` 侧记录/显示的文件名。运行时编译行为本身不变(只改路径归属判断);依赖 `__closure__` 结构是针对当前 torch 版本 wrapper 形态的实现细节,若未来 PyTorch 改变 callable 包装方式,该回退查找可能再次失效(回退失败时保持原行为,只是路径仍不正确,不会崩溃)。作者验证方式为本地 serve Llama-3.2-1B-Instruct 检查文件名形态。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
