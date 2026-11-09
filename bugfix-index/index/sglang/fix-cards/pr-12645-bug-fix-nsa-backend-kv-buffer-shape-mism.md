# PR #12645: [Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12645](https://github.com/sgl-project/sglang/pull/12645) |
| 作者 | Johnsonms |
| 合并日期 | 2025-11-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
NSATokenToKVPool 计算 override_dim 时 rope 维按 fp8(1 字节)而非实际 bf16(2 字节)计,DeepSeek-V3.2 NSA 后端 KV buffer 形状不匹配;改用 rope_storage_dtype 修复。

## 问题现象
DeepSeek-V3.2 使用 NSA 后端时 KV buffer 形状不匹配(标题:"Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2")。正文未展开具体报错信息与触发日志,完整现象见关联 issue #12643。

## 根因
正文披露:`NSATokenToKVPool.__init__()` 中 `override_kv_cache_dim` 的计算对 rope 维存储使用了错误的 dtype 大小——用了 `dtype.itemsize`(KV 缓存为 fp8 时是 1 字节),但 rope 维实际以 bfloat16 格式(2 字节)存储,并未被量化到 fp8。于是为 fp8 KV 计算的每 token 覆盖维度偏小,与实际写入布局不符,导致 KV buffer 形状不匹配。

## 修复方案
改 `python/sglang/srt/mem_cache/memory_pool.py` 的 `NSATokenToKVPool`:

1. 新增类属性 `rope_storage_dtype = torch.bfloat16`,显式记录 rope 维存储格式;
2. `override_dim` 计算中把 `qk_rope_head_dim * dtype.itemsize` 改为 `qk_rope_head_dim * self.rope_storage_dtype.itemsize`;
3. 补注释说明 "rope dimension is stored in original dtype (bf16), not quantized to fp8"。

```diff
+    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16
...
         override_dim = (
             kv_lora_rank
             + kv_lora_rank // self.quant_block_size * 4
-            + qk_rope_head_dim * dtype.itemsize
+            + qk_rope_head_dim * self.rope_storage_dtype.itemsize
         )
```

验证:作者运行 `python test_deepseek_v32_basic.py` 与 `python test_deepseek_v32_nsabackend.py`,并附了 benchmarking 截图(正文为图片链接,无文字数值)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 `NSATokenToKVPool` 的 override_dim 计算(fp8 KV 存储 + NSA 的 DeepSeek-V3.2 场景)。每个 KV token 的每维字节数按 rope bf16 实情修正(每 token 增大 `qk_rope_head_dim` 字节),影响该池的显存分配与 buffer 形状;非 NSA 或非 fp8-KV 路径不走该计算,不受影响。

## 关联
- https://github.com/sgl-project/sglang/issues/12643

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
