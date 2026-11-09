# PR #47614: [Fix] PP: record received proxy tensors on the forward stream

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47614](https://github.com/sgl-project/sglang/pull/47614) |
| 作者 | Phoenix3334 |
| 合并日期 | 2026-11-02 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
异步 PP 解码中接收的 proxy 张量被分配器提前复用、污染输出,run_batch 前 record_stream 锁定生命周期。

## 问题现象
异步 PP decode(`--pp-async-batch-depth > pp_size / 2`)下,大量请求连续输出 token 0(Kimi-K3 PP2×TP4 depth 2:192 个请求中 51-98 个受影响)。这是输出被静默污染的正确性问题:未修复的 run 中,接收后与前向前的 checksum 比对在约 7500 次中有 76 与 378 次不一致;修复后不再出现。对应 issue #47613。

## 根因
正文披露:异步 PP decode 中,proxy 张量在通信流上被接收分配,之后才由 forward 流消费;`event_loop_pp_disagg_decode` 在投递下一次 recv 前就丢弃上一个 Python 引用。当排队的 forward 足够多时,CUDA caching allocator 会把同一块显存分给下一次 recv,覆写尚未被排队 forward 读取的数据,从而破坏模型输出。即本质是分配器生命周期(allocator lifetime)竞态,不是流序问题。

## 修复方案
改动 `python/sglang/srt/managers/scheduler_pp_mixin.py` 的 `_pp_launch_batch`,在 `run_batch` 之前对每个非 CPU 的 proxy 张量调用 `record_stream(self.forward_stream)`,把存储生命周期保护到 forward 流用完为止:

```python
if pp_proxy_tensors is not None:
    # Prevent allocator reuse of received tensors while the forward
    # stream is still consuming them.
    for tensor in pp_proxy_tensors.tensors.values():
        if isinstance(tensor, torch.Tensor) and not tensor.is_cpu:
            tensor.record_stream(self.forward_stream)
result = self.run_batch(cur_batch, pp_proxy_tensors)
```

正文强调:不引入全局同步或新的执行序屏障,`record_stream` 记录的是分配器生命周期而非流顺序。所有 PP 事件循环都经过 `_pp_launch_batch`,共享此修复位置;第一阶段传 `None` 不受影响。另把 idle-counter 测试夹具的 `_pp_recv_proxy_tensors` mock 改为 `return_value=None` 以匹配真实接收路径。

## 影响与回归风险
- 影响范围(验证数据):Kimi-K3 8×B300(fake PD transfer)五种 PP/TP/DCP 组合下受影响请求全部归 0(如 PP2×TP4 depth 3:98/192 → 0/192);GSM8K(5-shot,200 问,PP2×TP4+DCP4 depth 2)修复前 0.985、修复后 0.980。性能:depth 0 下修复后慢 0.52-0.93%(小样本)。
- 影响窗口:正文说明 v0.5.22 与 main 在生命周期关键逻辑上一致,main 预期同样受此竞态影响;B300 复现与验证在 v0.5.22 完成,未在 main 上单独复现。
- 回归面:所有 PP 事件循环的 `_pp_launch_batch` 路径;对每个 GPU proxy 张量多一次 `record_stream` 调用,属轻量登记。正文明确这是正确性/生命周期修复而非性能优化,并限定真实两节点 PD、全部 PP 事件循环组合及 depth≥3 的普通 PP 仍未测试;专门通信流(`SGLANG_PP_COMM_OVERLAP=1`)的验证用的是早期等价实现而非最终 diff。审查中删除了只能 mock `record_stream`、无法复现分配器复用的单测。

## 关联
- [#47613](https://github.com/sgl-project/sglang/issues/47613)(被修复的 issue)
- [#47736](https://github.com/sgl-project/sglang/pull/47736)(mickqian 的 DCP CI 夹具修正,已入 main)
- [review 讨论](https://github.com/sgl-project/sglang/pull/47614#issuecomment-6567604516)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
