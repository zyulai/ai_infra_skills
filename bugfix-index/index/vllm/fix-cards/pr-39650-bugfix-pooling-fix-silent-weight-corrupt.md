# PR #39650: [Bugfix][Pooling] Fix silent weight corruption with buffer-reusing iterators

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39650](https://github.com/vllm-project/vllm/pull/39650) |
| 作者 | pedramr |
| 合并日期 | 2026-04-13 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`ModelForPooling.load_weights` 用 `(*seen_weights, *weights)` 提前耗尽迭代器,runai_streamer 复用 I/O buffer 时权重被静默改写;改为 `itertools.chain` 惰性迭代并对 probed 权重 `.clone()`。

## 问题现象
Pooling/embedding 模型在使用 `runai_streamer` 加载权重(设置 `RUNAI_STREAMER_MEMORY_LIMIT=0`)时,所有权重被静默破坏(正文:"silently corrupting all pooling/embedding model weights")——没有报错,加载照常完成,但模型参数内容已错。机理(正文 Summary):`ModelForPooling.load_weights` 中 `(*seen_weights, *weights)` 在生成器创建时就把整个权重迭代器立即(eagerly)耗尽;而 runai_streamer 在 `MEMORY_LIMIT=0` 下复用共享 I/O buffer,前一张张量的 buffer 在被真正消费前就被后一张覆盖,最终加载到模型里的是被覆盖后的错误数据。

## 根因
正文完整披露:① `mapped_weights` 生成器表达式 `(*seen_weights, *weights)` 会在创建时立即展开并耗尽整个 `weights` 迭代器,失去惰性;② runai_streamer 的 buffer 复用依赖调用方逐张消费,一旦被提前展开,共享 I/O buffer 在张量被消费前被覆写,造成静默损坏。此外 packed-weight 重映射路径(q/k/v_proj 合并为 qkv_proj)会先探测(probe)权重再做映射,被 probe 过的张量引用同一 buffer,同样会在复用中失效,需要克隆。

## 修复方案
`vllm/model_executor/models/adapters.py` 两处改动,并在 `tests/models/test_adapters.py` 新增两个回归测试(`test_pooling_load_weights_with_buffer_reuse`、`test_pooling_load_weights_clones_probed_weights`,均模拟 buffer 复用迭代器):
1. `seen_weights.append((name, loaded_weight.clone()))`——对已见权重克隆,使 packed-weight 重映射(q/k/v_proj → qkv_proj)在 buffer 复用下仍有效;
2. `(*seen_weights, *weights)` 改为 `itertools.chain(seen_weights, weights)`,惰性逐张消费。

```diff
-                seen_weights.append((name, loaded_weight))
+                # Clone because the iterator may reuse the tensor buffer
+                seen_weights.append((name, loaded_weight.clone()))
...
-                for name, weight in (*seen_weights, *weights)
+                for name, weight in itertools.chain(seen_weights, weights)
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:`vllm/model_executor/models/adapters.py` 的 `ModelForPooling.load_weights`,即所有经 pooling 适配器加载权重的 embedding/pooling 模型;`.clone()` 会增加等量的一次权重拷贝内存开销(已见权重列表);`itertools.chain` 改变消费时序(由 eager 变 lazy),对普通非复用迭代器行为等价。作者验证:两个新测试无修复时 FAIL、有修复时 PASS;仅做 chain 不做 clone 时第二个测试仍 FAIL;GPU 端到端 runai_streamer(`MEMORY_LIMIT=0`)加载 embedding 模型,输出与 ground truth 逐位(bit-for-bit)一致。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
