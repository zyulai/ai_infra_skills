# PR #20310: [tokenizer] improve non streaming request processing + some small fixes.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20310](https://github.com/sgl-project/sglang/pull/20310) |
| 作者 | alexnails |
| 合并日期 | 2026-04-10 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 35 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
非流式请求处理含 O(N^2) 操作;ReqState 改缓冲模式、仅在结束时产出响应,另附若干 tokenizer/detokenizer 小优化。

## 问题现象
非流式(non-streaming)请求处理效率低下:正文称其中最大的改动是把非流式请求处理改得更高效、避免 O(N^2) 操作,输出 token 越多开销越明显。作者写了小基准测试,显示「随 token 输出增长有清晰改善」(详细 bench_serving 矩阵见评论区附件)。正文未描述崩溃或错误输出。

## 根因
正文披露:非流式请求处理路径存在 O(N^2) 操作(具体机制由 diff 推断):此前每个输出批次都执行 `state.text += ...`(字符串重复拼接)与 `state.output_ids.copy()`,并把含完整文本/ids 的 `out_dict` 逐批 append 进 `out_list`,而非流式请求只需要最终一份结果,造成重复拷贝与二次方开销。

## 修复方案
- `tokenizer_manager.py`:`ReqState` 新增缓冲模式(`buffer_text` / `text_chunks` / `append_text` / `get_text` / `get_crash_dump_output`),经 `make_req_state` 工厂按请求是否流式决定缓冲;非流式请求文本按 chunk 缓冲、`"".join` 一次性产出:
```python
def append_text(self, chunk: str):
    if self.buffer_text:
        self.text_chunks.append(chunk)
    else:
        self.text += chunk
```
`_handle_batch_output` 中非流式且未结束时 `out_dict = None`,不再向 `out_list` 塞中间响应,仅结束时 push 并 `event.set()`;同时进一步尊重 `stream_output`(server_args 默认 False),`_wait_one_response` 检查 `self.server_args.stream_output`,失败时增加 dump,请求异常时断开连接。
- `detokenizer_manager.py`:batch_decode fast-path 把对 skip_list/space_list 的两次 `all(...)` 合并为一次 zipped 遍历;streaming chunk 更新改用真值判断与 `+=`。
- `async_dynamic_batch_tokenizer.py`:kwargs 批量判断由 `set(str(sorted(kw.items())))` 序列化比较改为直接 `all(kw == first_kw ...)`。
- 验证:bench_serving(Qwen3-1.7B,500 prompts,request-rate 50,input 128 / output 16384,流式与非流式两组);作者称准确性不受影响,计划在 H200 上再测。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:tokenizer_manager 的输出聚合与 `_wait_one_response` 逻辑(非流式路径重写,流式行为保持);detokenizer 的 batch decode fast-path 与流式 chunk 更新;动态批量 tokenizer 的 kwargs 比较。风险集中在非流式响应的组装时机(现在仅在 finished 时产出),以及 abort 输出改用 `state.get_text()`。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
