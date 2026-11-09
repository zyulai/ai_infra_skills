# PR #39423: [NPU][Bugfix] Disable pinned memory to fix DeepSeek-V2 DP-attention hang

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39423](https://github.com/sgl-project/sglang/pull/39423) |
| 作者 | iridiumine |
| 合并日期 | 2026-09-15 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
NPU 平台 pinned memory 未经验证导致 DeepSeek-V2 DP-attention 间歇性死锁,强制禁用恢复旧行为。

## 问题现象
DeepSeek-V2 在 DP-attention 推理中间歇性挂起(死锁)。两次独立复现:
- #36472 作者观察到开启 pin_memory 后 DeepSeek-V2 在正常 DP-attention 推理中 hang;
- 在 Ascend 910C 上以 DP-attention retraction 压力复现同样 hang(`TestDPRetract`,即 `SGLANG_TEST_RETRACT=1` + 较小 KV budget)。
两种场景在 `is_pin_memory_available=False` 时 hang 都消失。修复后在 NPU 上多次重跑 `TestDPRetract` 压力复现(DP attention + 4500-token KV budget + radix retract stress + regex/JSON 约束生成)全部通过。

## 根因
正文披露:#36472 引入 in-tree 的 `NPUSRTPlatform`,使 NPU 机器解析到它而不是基础 `SRTPlatform`。与基础默认(`is_pin_memory_available=False`)不同,新类返回 `True`,悄悄把 next-token-logits 拷贝、xgrammar bitmask 分配等热路径切换到 pinned memory + `non_blocking` H2D 拷贝。而本技术栈上 torch_npu 的 pinned-memory + 异步拷贝语义并未针对 CANN 验证过:`non_blocking` H2D 的正确性依赖一条尚未在 NPU 上建立的流序保证,在 DP attention 下(compute、copy、HCCL 通信流并发运行)推理会间歇性死锁。

## 修复方案
- `python/sglang/srt/platforms/npu.py`:强制 `is_pin_memory_available` 返回 `False`,恢复 #36472 之前的行为:

```python
def is_pin_memory_available(self, device=None) -> bool:
    return False
```

- `test/registered/unit/platforms/test_platform_interface.py`:同步单测断言,把原来对无参/`npu` device 的 `assertTrue` 改为 `assertFalse`,并加注释说明 pinned memory 在 NPU 上保持禁用,待针对 CANN 验证后再启用(可先挂在 debug 环境变量后)。

## 影响与回归风险
- 影响窗口:#36472 引入 in-tree `NPUSRTPlatform` 之后至本修复合并之间的 NPU 部署。
- 回归面:NPU 上所有原先走 pinned memory 的 H2D 拷贝路径(next-token-logits 拷贝、xgrammar bitmask 分配等)退回非 pinned 拷贝;正文指出 pinning 的每步收益仅限 KB 级 decode 路径 buffer,因此禁用是安全的,但理论上略有性能回退。平台接口单测断言随之更新。

## 关联
- [#36472](https://github.com/sgl-project/sglang/pull/36472)(引入 NPUSRTPlatform、改变 pin_memory 默认值的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
