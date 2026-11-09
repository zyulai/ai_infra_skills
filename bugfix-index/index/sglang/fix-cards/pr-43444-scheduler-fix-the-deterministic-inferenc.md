# PR #43444: [Scheduler] Fix the deterministic-inference hang when a prefill chunk is shorter than its alignment

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#43444](https://github.com/sgl-project/sglang/pull/43444) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-10-10 |
| labels | documentation, run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
确定性推理把每个 prefill chunk 截断到 4096 对齐倍数,per-rank chunk 小于对齐值时被截成 0 token,prompt 永远排队挂死;triton 改为 KV padding 对齐、flashinfer 改为启动时报错。

## 问题现象
开启确定性推理(deterministic inference)后,长 prompt 永远等待在队列里无法被调度(hang)。触发条件:per-rank 的 prefill chunk 小于对齐值(默认 4096)——要么 `--chunked-prefill-size` < 4096,要么 DP attention 把 chunk 拆到多 rank 导致每 rank 不足 4096。此时每个 chunk 都被截断为 0 个 token,长于 chunk 的 prompt 永远进不了调度。

## 根因
正文披露:deterministic inference 会把每个 prefill chunk 截断到对齐值(triton 下原为 `SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE=4096`)的整数倍;当 per-rank chunk 本身小于对齐值时,截断结果恒为 0 token,chunk 无法推进,导致排队永不结束。

## 修复方案
按后端分别处理:

- Triton:直接去掉 prefill chunk 的对齐截断。新增 `python/sglang/kernels/ops/attention/extend_attention.py::align_window_kv_to_tiles`,把每条请求的 sliding-window KV 列表用其首 key 的拷贝 pad 到绝对 128 倍数(tile 边界),使 chunked prefill 与 unchunked prefill 在相同的 tile 内求和、输出 bitwise 一致(每次 forward 构建一次,仅确定性推理启用;attention kernel 本身不变)。`triton_backend.py` 的 `ForwardMetadata` 增加 `aligned_window_kv_indptr/indices` 字段并在 `init_forward_metadata`/`_forward_extend_unified` 接入。
- Flashinfer 与 DSA kpool:保留对齐,但 per-rank chunk 短于对齐值时启动即抛 `ValueError`(decode server 豁免),而不是运行时挂死。`scheduler.py` 新增 `check_truncation_align_fits_chunk`,报错信息给出建议(把 `--chunked-prefill-size` 提到至少 `align * attn_dp_size`,或调低 `SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE`)。
- 废弃 `SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE`(`environ.py` 标记 `_DeprecatedEnv`,文档同步删除)。
- 新增 `test/registered/unit/managers/test_deterministic_chunk_align.py` 及 `test_triton_attention_kernels.py` 的 chunked-vs-unchunked sliding-window 逐位对比用例。

关键 diff 摘录(scheduler.py):

```diff
+    def check_truncation_align_fits_chunk(self):
+        """A chunk shorter than the alignment truncates to zero tokens, so a prompt
+        longer than the chunk would wait in the queue forever."""
+        align = self.truncation_align_size
+        chunk = self.chunked_prefill_size
+        if align is None or chunk is None or chunk >= align:
+            return
```

## 影响与回归风险
- 影响窗口:正文未说明;仅影响开启 deterministic inference 的部署,正文明确「No behavior change without deterministic inference」。
- 回归面:triton 后端确定性推理的 prefill 不再截断对齐,改为 KV 列表 padding,chunked 与 unchunked 输出应保持 bitwise 相等(新增测试保证);双向注意力的层保留未 pad 列表(diff 注释说明 padding 仅由 sliding-window mask 屏蔽)。flashinfer/DSA kpool 下原本能「跑起来但挂死」的配置改为启动失败,属行为变化(错误提前暴露);设置已废弃环境变量会收到 deprecation 提示。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
