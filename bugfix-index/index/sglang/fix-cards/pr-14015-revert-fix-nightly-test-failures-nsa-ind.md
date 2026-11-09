# PR #14015: Revert "Fix nightly test failures: NSA indexer dtype and CPP radix cache init"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14015](https://github.com/sgl-project/sglang/pull/14015) |
| 作者 | Fridge003 |
| 合并日期 | 2025-11-26 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
回滚 #13958:该修复改动导致 dpsk v32 nightly 测试挂掉,还原 NSA indexer dtype 与 CPP radix cache 初始化的原实现。

## 问题现象
#13958「Fix nightly test failures: NSA indexer dtype and CPP radix cache init」合入后,导致 dpsk v32(DeepSeek-V3.2)在 nightly 测试中挂掉,PR 正文给出失败 CI 链接(https://github.com/sgl-project/sglang/actions/runs/19694975166/job/56476095353?pr=13151)。正文未描述更具体的报错信息与触发条件,只要求作者「在失败的 nightly 测试里只改张量数据类型」。

## 根因
正文未披露根因,仅认定 #13958 的改动是 dpsk v32 挂掉的直接原因,故整体回滚。从 diff 看,#13958 的回滚涉及两处非纯 dtype 调整:
- `nsa_indexer.py` 的 `_get_logits_head_gate`:原修复把「先对 bfloat16 输入做 weights_proj 投影、再转 float32 缩放」改为「先把 x 转 float32 再投影」,改变了投影计算所用的精度路径;
- `radix_cache_cpp.py` 的 `__init__`:原修复调整了 `self.token_to_kv_pool_allocator` 与 `self.device` 的赋值顺序(先赋 allocator 再取 device)。
回滚后两处均恢复到 #13958 之前的实现。修复本身反而打破 dpsk v32,推测与这两处行为变更有关(由 diff 推断)。

## 修复方案
整体 revert #13958,涉及两个文件:
- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`:`_get_logits_head_gate` 恢复原实现:

```python
weights, _ = self.weights_proj(x.float())
weights = weights * self.n_heads**-0.5
weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
```

- `python/sglang/srt/mem_cache/radix_cache_cpp.py`:恢复原赋值顺序,`self.device = self.token_to_kv_pool_allocator.device` 在前、`self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator` 在后。

## 影响与回归风险
- 影响窗口:正文未说明(介于 #13958 合入与本回滚合并之间)。
- 回归面:影响 NSA(Native Sparse Attention)indexer 的 logits head gate 计算精度路径,以及 CPP radix cache 的初始化逻辑;#13958 想修的 nightly 测试问题会随之恢复存在。

## 关联
- [#13958](https://github.com/sgl-project/sglang/pull/13958)(被回滚的修复)
- [dpsk v32 失败 CI run](https://github.com/sgl-project/sglang/actions/runs/19694975166/job/56476095353?pr=13151)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
