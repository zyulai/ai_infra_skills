# PR #64986: [Bugfix][Attention] Fix KV-sharing fast prefill hang from unsynchronized host offsets

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64986](https://github.com/vllm-project/vllm/pull/64986) |
| 作者 | hojuna |
| 合并日期 | 2026-11-06 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论

async 调度下 fast prefill 的 host 偏移未同步就读,解码内核拿到垃圾长度挂死,改为 arange/阻塞拷贝。

## 问题现象

开启 `--kv-sharing-fast-prefill` 且使用 async 调度时,服务挂死。作者端到端复现:B200、Gemma 4 E4B fine-tune(bf16)、`vllm serve --kv-sharing-fast-prefill --enable-prefix-caching -cc '{"cudagraph_mode":"PIECEWISE"}'`、MRV2、async scheduling,重放 `/v1/completions` 请求(max_tokens=1,96 并发):4,649 请求切片 4/4 次全部挂死,修复后 20 秒跑完;完整 31,649 请求基线挂死,修复后 105 秒完成。bug 在 MRV1/MRV2 以及 PIECEWISE、FULL_AND_PIECEWISE 两种 cudagraph 模式下都能复现。

## 根因

按正文披露:`decode_query_start_loc` 在 GPU 上计算,host 拷贝用的是 `.to("cpu", non_blocking=True)`;而 attention metadata 构建器(`split_decodes_and_prefills`、FlashInfer 的 `_compute_decode_query_lens`)在 host 上立即读取它。async 调度时该拷贝排在仍在执行的 GPU 工作之后,host 读到的是未写完的内存——`query_start_loc_cpu` 内容是垃圾,trtllm-gen 解码内核拿到错误的 query 长度后挂起。

## 修复方案

修改 `vllm/v1/attention/backends/utils.py` 的 `make_kv_sharing_fast_prefill_common_attn_metadata`,分两个分支保证 `query_start_loc_cpu` 返回时一定是最终值:

- 每请求一个 logit(无投机解码,`decode_max_query_len == 1` 且 `total_num_decode_tokens == num_reqs`):偏移就是 `[0, 1, ..., num_reqs]`,直接在 host 上用 `torch.arange(..., device="cpu")` 构建,公共路径零同步;
- 其他情况(投机解码 + fast prefill,如 #63553):改用阻塞拷贝 `decode_query_start_loc.cpu()`(在 `gpu_sync_allowed()` 内,留有 TODO),此时正确性优先于免同步。

作者提到免同步的替代方案是把 host 侧每请求 logits 计数从 model runner 传入(MRV2 的 `FastPrefillHelper.prepare` 已有 `cu_num_logits_np`),但会触及 `CommonAttentionMetadata` 与 runners,留作后续。新增回归测试 `tests/v1/attention/test_kv_sharing_fast_prefill_metadata.py`,CUDA 用例在保持 stream 忙碌、不先同步的情况下读取 host 偏移。

关键 diff 摘录:

```diff
+    if decode_max_query_len == 1 and total_num_decode_tokens == num_reqs:
+        decode_query_start_loc_cpu = torch.arange(
+            num_reqs + 1, dtype=decode_query_start_loc.dtype, device="cpu"
+        )
+    else:
+        with gpu_sync_allowed():
+            decode_query_start_loc_cpu = decode_query_start_loc.cpu()
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:仅 `--kv-sharing-fast-prefill` 的元数据构建路径。单 logit 路径改为纯 host 构建,无 GPU 拷贝;多 logit(投机)路径新增一次阻塞同步,可能略增该路径延迟。与 #60445 的单 logit `arange` 实现有重叠,后落地一方可去掉重复分支;#60445 的 multi-logit fallback 仍用 `non_blocking=True`,存在同样竞态,本 PR 未改它。

## 关联

- [issue #64985](https://github.com/vllm-project/vllm/issues/64985)(本 PR 修复的 issue,含定位插桩)
- [PR #60445](https://github.com/vllm-project/vllm/pull/60445)(重叠实现,multi-logit fallback 仍有同款竞态)
- [PR #63553](https://github.com/vllm-project/vllm/pull/63553)(投机解码 + fast prefill 场景)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
