# PR #50234: [Fix] weights: Reject mismatched manifests before distributed broadcasts

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#50234](https://github.com/sgl-project/sglang/pull/50234) |
| 作者 | shub-garg |
| 合并日期 | 2026-11-15 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
分布式权重更新 manifest 长度不等被 zip 静默截断、缺收仍报成功推进版本;接收前预检三列表长度并先备齐全部 buffer 再 broadcast。

## 问题现象
RL rollout 权重同步:普通接收路径以 `zip(names, dtypes, shapes)` 迭代,列表长度不等时静默截断接收集合;scheduler 加载该前缀、报告成功并记录请求的版本——可能一边广播新 policy 一边部分权重仍属旧 policy。另一类:后面的 dtype/shape/allocation 非法时,更早的异步 broadcast 已投递,函数退出不等待这些 handle。CPU probe 证据(同一 probe 对照 upstream `4f8b6867534c8a6632845353997555fca857dd63` 与本分支,调用真实 scheduler、distributed receiver 与 weight loader,两权重 CPU 模型、受控 broadcast 边界,初值均 1,成功接收应得 w1=9/w2=10):"Two names, one dtype"、"Two names, one shape"、"Two names, no dtypes" 三个畸形长度场景 upstream 全部假成功(记录 v2,w2 仍是 1 或两权重都没变),本 PR 全部 Failure、零 receive、版本不变;非法第二个 dtype/shape 时 upstream 在投递一个 receive 且零 handle 等待后才失败,本 PR 在投递任何 receive 前失败;合法两权重更新两树结果与收到的值相同。正文注明:这些是确定性注入的 CPU 用例,不是测得的生产故障率或 GPU 输出结果。

## 根因
正文披露:接收路径对 manifest 长度无预检,依赖 zip 的隐式截断;buffer 分配与异步 broadcast 交错进行,后面条目的 dtype/shape/allocation 错误发生时前面的 async broadcast 已 post,退出时未 wait 这些 handle。

## 修复方案
`python/sglang/srt/model_executor/model_runner_components/weight_updater.py` `receive_weights_from_distributed`:入口处在分配或分派到任一接收格式之前校验
```python
if not len(names) == len(dtypes) == len(shapes):
    raise ValueError(
        "Weight update metadata must have equal lengths: "
        f"names={len(names)}, dtypes={len(dtypes)}, shapes={len(shapes)}"
    )
```
普通路径改为两阶段:先循环把全部 buffer 准备完(`torch.empty`,让 PyTorch 的 dtype/shape/allocation 校验先于任何数据 collective 完成),再以 list comprehension 统一 post `torch.distributed.broadcast(async_op=True)` 并逐个 wait。保留 dtype 别名与 `torch.dtype` 输入、scalar 与零尺寸 shape、张量顺序、flattened 格式与空 manifest 的 no-op;不加新 endpoint、option、wire field 或 collective。测试扩展 `test/registered/unit/model_executor/test_weight_update_tensor_rank.py`:修复前 focused run 15 个回归用例失败,修复后完整 focused 文件 6 tests + 25 subtests 全过;更广的 scheduler/weight/KV 套件 64 tests + 39 subtests passed;与 #50229、#50230 合测 278 tests + 195 subtests passed, 14 skipped;9 个两进程 CPU Gloo 场景(含被拒轮次后接合法更新)通过。

## 影响与回归风险
- 影响窗口:正文未给出。
- 回归面:`receive_weights_from_distributed` 的普通与 `flattened_bucket` 两条接收格式都新增长度预检;第一个 broadcast 现在始于全部 buffer 分配之后,改变权重更新控制面上的分配/通信重叠(GPU 更新延迟与 serving 吞吐未基准);恢复限制:接收侧预检无法取消已进入 collective 的 trainer——这类 sender 仍需自行观察失败并恢复传输;本修复只阻止静默前缀接受并避免本接收方开启不完整的一轮,不加 sender handshake、rollback 或跨 rank 恢复协议;硬件 CI 与 GPU 延迟/精度尚待验证(正文 checklist 注明)。

## 关联
- [#50229](https://github.com/sgl-project/sglang/pull/50229)(互补:传播 TP 更新结果)
- [#42271](https://github.com/sgl-project/sglang/pull/42271)(external receivers,明确不改默认 native 路径)
- [#48276](https://github.com/sgl-project/sglang/pull/48276)(intentional custom-loader refusal)
- [#50230](https://github.com/sgl-project/sglang/pull/50230)(合测对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
