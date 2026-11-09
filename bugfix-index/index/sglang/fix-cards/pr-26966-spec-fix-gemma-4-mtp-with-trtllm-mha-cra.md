# PR #26966: [Spec] Fix Gemma 4 MTP with `trtllm_mha` crash issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26966](https://github.com/sgl-project/sglang/pull/26966) |
| 作者 | kpham-sgl |
| 合并日期 | 2026-06-02 |
| labels | blackwell, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
trtllm_mha 在 init 缓存 SWA 池,FROZEN_KV MTP 下取到 draft 非 SWA 池致越界读;改为从 allocator 解析修复。

## 问题现象
在 hybrid-SWA 模型(如 Gemma-4)上使用 `trtllm_mha` + FROZEN_KV MTP(NEXTN)投机解码时,一旦 SWA pool 填充量超过 SWA 层与全量层的比例,就会触发 CUDA illegal memory access 崩溃。作者在 B200(SM100)上用端到端脚本 repro_26957_e2e.py 复现:修复前必崩,修复后通过。

## 根因
正文披露:`TRTLLMHAAttnBackend` 在 `__init__` 时从 `model_runner.token_to_kv_pool` 缓存 SWA 状态。在 draft worker 上,这个字段是 draft 自己的非 SWA 池,于是 `use_sliding_window_kv_pool` 为 `False`,SWA→full 池的页索引翻译变成死代码——SWA 层拿到的是未翻译的 full-pool 原始索引,去读容量更小的 SWA k-cache 时越界。

## 修复方案
改 `python/sglang/srt/layers/attention/trtllm_mha_backend.py`:
- 删除在 `__init__` 用 `token_to_kv_pool` 判断并缓存 SWA 池的做法,新增静态方法 `_resolve_swa_kv_pool`,改从 `model_runner.token_to_kv_pool_allocator` 的 `get_kvcache()` 解析 `SWAKVPool`。理由(正文):FROZEN_KV MTP 中 draft 的 `token_to_kv_pool` 每次 forward 都会被换成 target 的 `SWAKVPool`,而 `token_to_kv_pool_allocator` 在 init 时设置一次、保持稳定,是正确的事实来源;该做法也与 FlashInfer backend 已有的按 allocator 判定 SWA 的逻辑一致;
- 移除 `use_sliding_window_kv_pool` 布尔标志,`_maybe_translate_swa`、`_alloc_swa_page_table`、`_get_layer_cache_loc` 改为以 `self._swa_kv_pool is None / is not None` 判断;
- 正文另说明:不把 Gemma-4 draft 注册为 hybrid-SWA——draft 自身无 KV(frozen),其自己的 `SWAKVPool` 会按 draft 层 id 键控而 attention 用的是重映射后的 target 层 id(会 KeyError 且浪费分配);allocator 已指向正确的 target 池。

关键摘录:
```diff
-        kv_pool = model_runner.token_to_kv_pool
-        self.use_sliding_window_kv_pool = isinstance(kv_pool, SWAKVPool)
+        self._swa_kv_pool: Optional[SWAKVPool] = self._resolve_swa_kv_pool(model_runner)
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围(与引入 trtllm_mha SWA 支持之后相关)。
- 回归面:改动仅触及 trtllm_mha backend 的 SWA 池解析与判空逻辑。普通 hybrid-SWA 模型(非 MTP)路径同样改为按 allocator 解析,行为应等价但来源更稳;`getattr` 容忍 attention 测试夹具使用的最小 allocator stub;涉及 FROZEN_KV MTP draft worker 的所有 trtllm_mha SWA 翻译路径。

## 关联
- https://github.com/sgl-project/sglang/issues/26957
- https://github.com/sgl-project/sglang/pull/26531
- 复现脚本: https://gist.github.com/kpham-sgl/d696a3b8c2fdc959c940f7a8afdd9c9b

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
