# PR #14224: [bug fix] fix ima with get_mla_kv_buffer_kernel overflow

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14224](https://github.com/sgl-project/sglang/pull/14224) |
| 作者 | XucSh |
| 合并日期 | 2025-12-04 |
| labels | run-ci |
| 热度 | 👍 6 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论

DeepSeek3.1 大 KV buffer 下 MLA triton kernel 的 `loc` 按 int32 参与指针偏移导致溢出 IMA;把 `loc` 转成 int64 修复。

## 问题现象

在 H20 上以 DeepSeek3.1 触发 IMA(illegal memory access)。复现配置:2 节点、`--tp 1 --pp-size 16`、`SGLANG_PP_LAYER_PARTITION="3,3,4,...,3"`、`--attention-backend fa3 --chunked-prefill-size 6144 --mem-fraction-static 0.8 --disable-radix-cache`、`CUDA_LAUNCH_BLOCKING=1`;客户端用 bench_serving 以 512 并发、random-input 13107、random-output 1、request-rate 10 打流。调度器报错栈最终落在 `memory_pool.py` 的 `get_mla_kv_buffer_triton` → `get_mla_kv_buffer_kernel[grid](...)` 启动处:`RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`。由 @whybeyoung 报告。

## 根因

正文 Modifications 写明修复方式是「extend loc to int64」,标题指明问题是 `get_mla_kv_buffer_kernel` 溢出(overflow),但未逐步披露溢出机制。(由 diff 推断)triton kernel 中 `loc = tl.load(loc_ptr + pid_loc)` 加载出的位置索引默认为 int32,后续以 `loc * buffer_stride` 计算 KV buffer 指针偏移;在 DeepSeek3.1 MLA 这类超大 KV buffer(配合 13107 长输入的 loc 值)场景下乘积超出 int32 表示范围,指针越界导致 IMA。

## 修复方案

改 `python/sglang/srt/mem_cache/utils.py`,把两个 kernel 中加载的 loc 都转换为 int64(`set_mla_kv_buffer_kernel` 与 `get_mla_kv_buffer_kernel` 各一处):

```diff
-    loc = tl.load(loc_ptr + pid_loc)
+    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
     dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs
```

```diff
     pid_loc = tl.program_id(0)
-    loc = tl.load(loc_ptr + pid_loc)
+    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
     loc_src_ptr = kv_buffer_ptr + loc * buffer_stride
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围;受影响场景为 loc 值与 MLA KV buffer stride 乘积超过 int32 的大模型/大 buffer 配置(如 DeepSeek3.1 + PP 大规模部署),小模型通常不触发。
- 回归面:改动仅涉及两个 triton kernel 内 loc 的 dtype,数值语义不变、仅扩大可表示范围;读写 MLA KV buffer 的 set/get 两条路径都会经过改动点,性能上 int64 索引开销可忽略。其他非 MLA kernel 未改。

## 关联

正文未列出 issue/PR 链接(仅 @ 提及 whybeyoung)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
