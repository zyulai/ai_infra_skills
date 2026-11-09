# PR #33517: Fix NaN logits from deterministic Triton extend on the unified memory pool

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33517](https://github.com/sgl-project/sglang/pull/33517) |
| 作者 | ch-wan |
| 合并日期 | 2026-08-10 |
| labels | ready-to-merge |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

三个 flag 组合(unified memory + triton + deterministic)下 extend 读 KV 虚拟/物理 id 混用产生 NaN;复用已算好的物理 id 修复。

## 问题现象

`--enable-unified-memory` + `--attention-backend triton` + `--enable-deterministic-inference` 三者同时启用时产生 NaN logits;任意两个组合都正常,因此现有测试均未覆盖。在 #33362 添加 PD 分离测试时发现,属 main 上的既有缺陷、与 PD 无关(#33362 靠不一起设这三个 flag 绕过)。失败比崩溃更隐蔽:只有 CI 武装 `SGLANG_ENABLE_ASYNC_ASSERT` 时才会中止(`_assert_async_cuda_kernel: Assertion 'NaN detected! sampler: next_token_logits' failed.`),否则运行正常完成、返回有限的 top-k logprob,本地损坏是静默的。

## 根因

正文披露:`--enable-deterministic-inference` 会把 Triton extend 路由到 `TritonAttnBackend._forward_extend_unified`——一个 1-stage kernel,前缀与 extend 两半 KV 都从 pool 读取(默认 2-stage 路径的 extend 半取自 `k`/`v` 参数,只从 pool 读前缀)。该 kernel 被喂了未翻译的 `forward_batch.out_cache_loc`:在 unified memory pool 上它是**虚拟** id,而前缀 `kv_indices` 已在 `init_forward_metadata` 翻译为物理 id。于是前缀按物理 id 读取、刚写入的 extend token 按虚拟 id 读取,错配读出垃圾,到 LM head 处变成 NaN。这也精确解释了三条件:无 deterministic 时 2-stage kernel 不从 pool 读 extend;无 unified memory 时 `out_cache_loc` 本就是物理 id;无 Triton 时不走该 kernel。

## 修复方案

`python/sglang/srt/layers/attention/triton_backend.py` 的 `_forward_extend_unified`:把 extend 读索引换成每次 forward 已为 KV 写入算好的翻译 `forward_metadata.out_cache_loc_full_physical`(与几行之上 `k is None` 分支的既有写法一致,line 1600 只是漏了)。SWA 层仍用原始虚拟 id(`translate_loc_from_full_to_swa` 契约上消费虚拟 id);复用每 forward 一次的翻译而非重算(该 kernel 每层跑一次);对静态池无影响(`out_cache_loc_full_physical` 为 `None`,translate 函数仅在 unified allocator 中定义)。附带修复了 deterministic Triton extend 下 unified SWA 池全注意力层的同路径问题。

```diff
+        elif self.forward_metadata.out_cache_loc_full_physical is not None:
+            extend_kv_indices = self.forward_metadata.out_cache_loc_full_physical
```

新增回归测试 `test/registered/attention/test_unified_memory_deterministic.py`(刻意钉死 Triton 后端),断言 unified 与 static 池逐 token 一致;验证其在未修代码上失败(1 个 NaN 断言)、修复后通过。修复后 `yujiepan/kimi-linear-tiny-random` 257-token prompt 上 unified 运行与 static 池 bit-identical(`[-11.908, -11.918, -11.880, -11.880]`)。

## 影响与回归风险

- 回归面:`TritonAttnBackend._forward_extend_unified` 的 extend KV 读索引;仅当 unified 分配器提供虚拟→物理翻译时生效,静态池路径不变。SWA 层分支保持虚拟 id。该测试在 CI 上验证修复前失败、修复后通过。

## 关联

- #33362(发现该缺陷的 PD 分离测试 PR,以不设三 flag 组合绕过)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
