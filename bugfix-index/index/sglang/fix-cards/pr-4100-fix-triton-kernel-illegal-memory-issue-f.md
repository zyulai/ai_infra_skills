# PR #4100: Fix triton kernel illegal memory issue for eagle

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4100](https://github.com/sgl-project/sglang/pull/4100) |
| 作者 | ispobock |
| 合并日期 | 2025-03-05 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
EAGLE 场景 extend_prefix_lens 长度与 batch size 不等,triton kernel 高负载下非法内存访问;加 has_prefix 判断修复。

## 问题现象
在 EAGLE 投机解码的 draft extend 与 target verify 阶段,triton kernel 出现 illegal memory issue。触发条件为高 workload(正文表述为 "memory is not enough")。正文未给出具体报错日志与影响版本。

## 根因
正文披露:draft extend 与 target verify 阶段 `extend_prefix_lens` 未被准备(unprepared),其长度不等于 batch size;triton kernel 按 batch size 逐行加载该张量,导致越界访问,高负载下表现为非法内存访问。

旧代码通过在 `init_new` 中对 `speculative_algorithm == "EAGLE"` 特判、跳过 triton position 计算来规避此问题(diff 中被删除的 TODO 注释 "Fix triton kernel illegal memory access for EAGLE" 亦与此印证)。

## 修复方案
改动集中在 `python/sglang/srt/model_executor/forward_batch_info.py`:

1. 删除 `init_new` 中对 EAGLE 的特判,使 EAGLE 场景恢复走 triton 路径(`compute_position_triton`)。
2. `compute_position_triton` 新增 `has_prefix = extend_prefix_lens.shape[0] == batch_size` 并传入 kernel。
3. `compute_position_kernel` 新增 `has_prefix: tl.constexpr` 参数,无 prefix 时不加载该张量:

```diff
-    prefix_len = tl.load(extend_prefix_lens + pid)
+    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
     seq_len = tl.load(extend_seq_lens + pid)
```

即当 `extend_prefix_lens` 长度与 batch size 不一致时,kernel 不再读取该张量,prefix 长度按 0 处理,消除越界。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:该文件中所有非 `torch_native` attention backend 的 extend 前向都会经过 `compute_position_triton`,本次改动改变其行为分支:
  - EAGLE 场景从此前"跳过 triton 路径"变为走 triton 路径 + `has_prefix=False` 分支(prefix 按 0 计算),性能路径与数值路径均发生变化;
  - 正常预填充(有 prefix 且长度等于 batch size)走 `has_prefix=True`,逻辑不变;
  - `has_prefix` 为 constexpr,按块编译,无运行时额外开销。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
