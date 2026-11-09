# PR #34183: [Bugfix][Core] Fix CPU memory leak from Request reference cycle in prefix caching

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34183](https://github.com/vllm-project/vllm/pull/34183) |
| 作者 | ywang96 |
| 合并日期 | 2026-02-10 |
| labels | bug, ready, v1 |
| 热度 | 👍 2 · 💬 3 |
| 分类 | KV Cache / Memory |
| bug 类型 | resource |

## 一句话结论
Request 经 partial 绑定 block_hasher 形成引用循环,多模态下 CPU 内存无界增长;改存未绑定函数修复。

## 问题现象
多模态模型推理时 EngineCore 的 CPU 内存(RSS)随请求轮次持续增长、无上界。正文给出实测:main 分支跑 5 轮,RSS 从 idle 3.67 GB 涨到 18.08 GB(R2-5 平均 +1.74 GB/轮);本 PR 修复后 5 轮仅 10.25 GB(R2-5 平均 +0.20 GB/轮)。正文提供复现脚本 test.py,确认修复有效。文本-only 推理因每个 Request 数据量小,基本无感。

## 根因
正文披露:#24964、#27896 为减少 GC 跟踪对象、降低 gen-2 回收频率做了改动;但多模态模型下每个 `Request` 可携带最高数十 MB 的 `mm_features`,一旦 Request 无法及时回收,内存就无界增长(FIXES #28726)。diff 新增注释进一步指明机制:构造时用 `partial(block_hasher, self)` 保存 hasher,形成 `Request -> partial -> Request` 引用循环,阻止引用计数立即回收该对象(由 diff 推断)。

## 修复方案
`vllm/v1/request.py`:
- 不再保存绑定 self 的 partial,改为 `self._block_hasher = block_hasher`(存未绑定函数)。
- 新增 `update_block_hashes()` 方法,内部调用 `self._block_hasher(self)` 并把新满块哈希追加到 `self.block_hashes`;`append_output_token_ids` 改调该方法。

关键 diff 摘录:
```diff
-        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
-        if block_hasher is not None:
-            self.get_hash_new_full_blocks = partial(block_hasher, self)
-            self.block_hashes = self.get_hash_new_full_blocks()
+        # Store the block hasher without binding self to avoid creating a
+        # reference cycle (Request -> partial -> Request) that prevents
+        # immediate garbage collection via reference counting.
+        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
+        self.update_block_hashes()
```
同步改动:`vllm/v1/core/sched/scheduler.py` 的 `_update_request_as_session` 改调 `session.update_block_hashes()`;`tests/v1/core/test_async_scheduler.py` 同步更新。

## 影响与回归风险
- 影响窗口:由 #24964、#27896 引入之后(正文未给具体版本)。
- 回归面:V1 prefix caching 的块哈希计算路径(`Request.append_output_token_ids`、多轮会话 `_update_request_as_session`),以及引用该接口的调用方;`get_hash_new_full_blocks` 属性被移除,任何外部引用需同步更新。

## 关联
- Fixes [#28726](https://github.com/vllm-project/vllm/issues/28726)
- [#24964](https://github.com/vllm-project/vllm/pull/24964)
- [#27896](https://github.com/vllm-project/vllm/pull/27896)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
