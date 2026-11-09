# PR #51667: [Bugfix] Fix cross-batch buffer race corrupting DiskBackend loads

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51667](https://github.com/vllm-project/vllm/pull/51667) |
| 作者 | fjosw |
| 合并日期 | 2026-09-02 |
| labels | bug, ready |
| 热度 | 👍 3 · 💬 9 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`_do_load` 返回前不清空 H2D 事件,下一批 preadv 覆写仍被读取的缓冲,补 synchronize 修复 KV cache 静默损坏。

## 问题现象
使用 #49644 新合入的 DiskBackend 做 KV offloading 时,跨批次的 load 存在静默数据损坏:重新加载(reload)时 GPU 上的 KV cache 内容可能被写坏。作者在审阅 DiskBackend 代码时发现该问题,正文未给出用户侧日志;回归测试的失败信息描述了症状——`preadv overwrote slot(s) [0, 1] while a prior batch's H2D was still reading them`,即前一批的 H2D 拷贝还在读缓冲时,下一批的 preadv 已把槽位覆写。

## 根因
正文披露:`_do_load` 的双缓冲流水线中,pending 的 H2D 事件记录在一个**每次调用局部**的列表里,并且与 `_do_store` 不同,返回前从不 drain(等待完成)。于是上一次 `_do_load` 调用尾部仍在进行的 H2D 拷贝还占着缓冲槽,后续 `_do_load` 的 preadv 就把同一 slot 覆写,前一批正在读的数据被污染,GPU KV cache 在 reload 时被静默破坏。这是典型的生命周期/同步缺失,不是计算逻辑错误。

## 修复方案
改 `vllm/v1/simple_kv_offload/disk_backend.py` 的 `_do_load`,在返回前同步所有未完成的 DMA 事件,与 `_do_store` 的既有模式对齐:

```diff
             ev = torch.Event()
             ev.record(stream)
             prev_dma_events[buf_slot] = ev
+
+        for ev in prev_dma_events:
+            if ev is not None:
+                ev.synchronize()
```

另新增回归测试 `test_do_load_drains_inflight_before_next_batch`,该测试在 main 上失败、带补丁通过;`pytest tests/v1/simple_kv_offload/test_disk_backend.py -v` 全部 3 个测试通过。作者注明 AI 辅助、逐行人工复核。

## 影响与回归风险
- 影响窗口:自 #49644 引入 DiskBackend(KV offloading)合入起,至本修复前;仅使用磁盘后端 KV offloading 且发生跨批次连续 load 的部署受影响。
- 回归面:按 diff 只在 `_do_load` 末尾增加 4 行事件同步,不改动缓冲分配、slot 计算或 store 路径。代价是 `_do_load` 返回会等待尾部 H2D 完成,理论上牺牲少量流水线重叠换取正确性;对正确性敏感场景是净收益。

## 关联
- [#49644](https://github.com/vllm-project/vllm/pull/49644)(引入 DiskBackend 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
