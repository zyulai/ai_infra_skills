# PR #18607: [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18607](https://github.com/sgl-project/sglang/pull/18607) |
| 作者 | 1am9trash |
| 合并日期 | 2026-02-12 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
AMD TP4 跑 dpsk-r1-fp4 + MTP 精度错误;通过 aiter 后端按 head 数选择内核模式并升级 aiter 修复。

## 问题现象
正文未描述具体现象,仅 Motivation 一句:「Fixed dpsk-r1-fp4 accuracy issue on tp4 with mtp」,即 AMD 上以 TP4 + MTP(EAGLE 投机解码)运行 DeepSeek-R1 系列模型时出现精度问题。正文给出复现/验证用 server 启动命令:aiter attention backend、fp8_e4m3 KV cache、EAGLE 3 步 topk1 4 draft tokens,并用 gsm8k 2000 题验证,修复后 Accuracy 0.928、Invalid 0.000。修复前的精度数值正文未给出。

## 根因
正文未披露根因,只说明修改内容是「aiter backend change in sglang」与「aiter update for tp4 fp8 prefill attn kernel (head=32)」(aiter 侧已合入 commit 6a0e7b26ccf33164785531212cc2ec2cde0b9243),即缺陷同时涉及 sglang 内 aiter 后端的内核选择逻辑与 aiter 上游的 TP4 fp8 prefill attention 内核(head=32)。

(由 diff 推断)aiter_backend.py 原逻辑对非 fp8 KV cache 一律关闭 mla_ps_kernel 并退回 non-persist/fast_mode;且 `make_mla_meta_data` 传入某布尔参数恒为 True。TP4 时 MLA head 数为 32,落入与 TP8(head=16)不同的内核形态,原选择策略不适用该形态,导致精度错误。修复后:head=32 强制 `fast_mode = True`、`intra_batch_mode = False`;仅当 `num_head == 16` 且非 fp8 KV cache 时才退回 non-persist;`make_mla_meta_data` 对应布尔参数由 True 改为 False。

## 修复方案
1. `python/sglang/srt/layers/attention/aiter_backend.py`:
   - `__init__` 中新增按 head 数分支:`if self.num_head == 32: fast_mode = True; intra_batch_mode = False`;
   - 关闭 persist MLA kernel 的条件收窄:`if self.kv_cache_dtype is not fp8_dtype` 改为 `if self.num_head == 16 and self.kv_cache_dtype is not fp8_dtype`(注释同步改为「for non-fp8 kv_cache on tp8」);
   - `make_mla_meta_data` 中 `nhead_kv` 后的一个位置参数由 `True` 改为 `False`。
2. `docker/rocm.Dockerfile`(两处):`AITER_COMMIT` 由 `v0.1.10.post2` 升级为 `v0.1.10.post3`,带入 aiter 上游对 tp4 fp8 prefill attn kernel (head=32) 的修复。

```diff
+            if self.num_head == 32:
+                fast_mode = True
+                intra_batch_mode = False
             # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
-            if self.kv_cache_dtype is not fp8_dtype:
+            if self.num_head == 16 and self.kv_cache_dtype is not fp8_dtype:
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:AMD 上 aiter attention backend 的 MLA 解码路径,尤其 TP4(head=32)与 TP8(head=16)下 persist/mla_ps/fast_mode/intra_batch_mode 的内核选择;非 fp8 KV cache 场景在 head≠16 时不再退回 non-persist,可能改变性能特征;`make_mla_meta_data` 参数变化影响所有走该元数据构造的 aiter MLA 调用;rocm 镜像构建固定到 aiter v0.1.10.post3。

## 关联
- aiter 上游修复 commit:https://github.com/ROCm/aiter/commit/6a0e7b26ccf33164785531212cc2ec2cde0b9243
- 其余 issue/PR:正文未列出(仅 cc @HaiShaw、@kkHuang-amd)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
