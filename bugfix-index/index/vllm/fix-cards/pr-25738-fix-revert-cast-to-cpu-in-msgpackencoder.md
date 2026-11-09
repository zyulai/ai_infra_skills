# PR #25738: fix: revert cast to cpu in `MsgpackEncoder._encode_tensor` to avoid hidden performance regressions

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25738](https://github.com/vllm-project/vllm/pull/25738) |
| 作者 | qthequartermasterman |
| 合并日期 | 2025-09-26 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
撤销 MsgpackEncoder 内隐式 .cpu() 设备搬运,改为 prompt embeds 预处理处显式转 CPU,避免隐藏性能回退。

## 问题现象
正文未描述具体故障现象(无复现日志);指出的是风险:尽管当时未观测到性能回退,但每次进程间序列化都做设备间 cast 非常昂贵,在生成关键路径上隐式发生,有引入重大性能回退的潜在可能。

## 根因
正文披露:PR #24278 在 `MsgpackEncoder` 中引入了序列化前把 tensor cast 到 CPU 的逻辑;正确做法(经 @njhill 建议)是让待 Msgpack 序列化的 tensor 在编码前显式位于 CPU,而不是在编码器内部隐式搬运。本 PR 是 #22962 的精神后继(后者在 OpenAI 兼容 API 层做了同样的 CPU 转换),并确保离线模式提交的 prompt embeds tensor 也先转 CPU。

## 修复方案
- `vllm/v1/serial_utils.py` 的 `_encode_tensor` 移除 `.cpu()`:

```diff
-        arr = obj.flatten().contiguous().cpu().view(torch.uint8).numpy()
+        arr = obj.flatten().contiguous().view(torch.uint8).numpy()
```

- `vllm/inputs/preprocess.py` 的 `_process_embeds` 中对 prompt_embeds 显式 `prompt_embeds = prompt_embeds.cpu()`,保证无隐藏设备传输进入生成关键路径。

## 影响与回归风险
- 影响窗口:正文说明自 #24278 引入 encoder 内 cast 起。
- 回归面:Msgpack 编码路径与 prompt embeds 预处理;调用方必须保证 tensor 在 CPU 后再序列化,否则 `_encode_tensor` 会在非 CPU tensor 上失败——把转移移到入口正是意图所在。

## 关联
[#24278](https://github.com/vllm-project/vllm/pull/24278)、[#22962](https://github.com/vllm-project/vllm/pull/22962)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
