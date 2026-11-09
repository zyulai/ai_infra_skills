# PR #6348: [fix] illegal memory in _fwd_kernel_ep_scatter_2 and _fwd_kernel_ep_gather

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6348](https://github.com/sgl-project/sglang/pull/6348) |
| 作者 | xutizhou |
| 合并日期 | 2025-05-18 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
大规模 EP DeepSeek prefill 重负载下 ep scatter/gather kernel 索引溢出导致非法内存访问;索引改用 int64 修复。

## 问题现象
大规模部署 EP DeepSeek 时,prefill 节点在重负载下偶发 illegal memory error(正文附报错截图)。触发条件为重负载、token 数达到数十万量级。正文提供复现脚本:`CUDA_LAUNCH_BLOCKING=1 python test.py`,构造 N=235234*3、D=7168、E=9、N_=304000*3 的张量后依次调用 `ep_scatter` 与 `ep_gather` 即可复现。

## 根因
正文披露:该错误由 `_fwd_kernel_ep_scatter_2` 中的索引溢出(index overflow)导致。

(由 diff 推断)两个 kernel 的循环变量、`tl.program_id`、`tl.atomic_add` 返回值等均以默认 int32 参与指针运算,`token_id * stride` 一类乘积在 token 数极大时超出 int32 表示范围,造成越界寻址;修复将参与地址计算的值统一转为 int64,与溢出机理吻合。

## 修复方案
改动集中在 `python/sglang/srt/layers/moe/ep_moe/kernels.py`,将索引计算提升为 int64:

1. `_fwd_kernel_ep_scatter_2`:循环变量 `token_id`、`topk_index` 及 `atomic_add` 返回的 `dest_token_index` 均先 `.to(tl.int64)` 再参与指针运算;存入 `output_index` 时使用 int32 中间值:

```diff
-    for token_id in range(start_token_id, total_token_num, grid_num):
+    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
+        token_id = token_id_int32.to(tl.int64)
```

2. `_fwd_kernel_ep_gather`:`tl.program_id(0)`、`tl.program_id(1)`、`cur_token`、`topk_index`、`source_token_index` 同样转为 int64 后再寻址。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:EP MoE 中所有经 `ep_scatter`/`ep_gather` 的调用路径(大规模 EP DeepSeek prefill 为主)。改动仅把参与地址计算的索引位宽从 int32 提升为 int64,数值语义不变;额外引入若干类型转换指令,性能影响取决于 kernel 实测;`output_index` 的存储值改用 int32 中间值再写入,写入数值范围与原实现一致。

## 关联
正文未列出(致谢 Ch-Wan <cwan39@gatech.edu> 定位根因并提供主要测试脚本,列为 co-author)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
