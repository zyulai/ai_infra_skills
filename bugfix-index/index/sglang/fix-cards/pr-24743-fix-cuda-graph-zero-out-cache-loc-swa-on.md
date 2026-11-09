# PR #24743: fix(cuda_graph): zero out_cache_loc_swa on pad and use int32 (hybrid-SWA accuracy fix)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24743](https://github.com/sgl-project/sglang/pull/24743) |
| 作者 | JoyFuture |
| 合并日期 | 2026-05-09 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
decode CUDA graph pad 时未清零 out_cache_loc_swa,残留索引把垃圾 KV 写进活跃请求的 SWA 槽,MiMoV2.5 Pro 精度回退最多 10 分,现 pad 时清零并改 int32。

## 问题现象
MiMoV2.5 Pro(经 `model_config.py` 中 `MIMO_V2_MODEL_ARCHS` 注册为 hybrid SWA)出现可测量的精度回退。同一 checkpoint、相同 eval 负载下:before/after 为 aime24_25 80.3→90.3(+10.0)、cmmlu 87.6→90.1(+2.5)、minerva_math 89.4→93.6(+4.2)、mmlu-pro 82.6→85.1(+2.5)、mmlu_redux 93.1→94.97(+1.87)、supergpqa 59.1→62.4(+3.3)。复现条件:decode CUDA graphs 开启,且实际 batch size 小到会被 pad 到某个 `capture_bs` 桶——即 decode 稳态的常见情况。

## 根因
正文披露两点:
1. 主 `CudaGraphRunner` 的 `populate_from_forward_batch` 在 `bs != raw_bs` 时只清零 `out_cache_loc`,`out_cache_loc_swa` 未清零,且随后的拷贝只写前 `raw_num_token` 项;而捕获的 graph 对整个 `out_cache_loc_swa[:num_tokens]` 切片执行 `set_kv_buffer`。于是 pad 区间 `[raw_num_token : num_tokens]` 保留上一次 replay 的真实 SWA 索引,pad token 把垃圾 KV 写进此前活跃请求正占用的 SWA 槽。#23552 之前的回退路径(逐层 `translate_loc_from_full_to_swa(out_cache_loc)`)无此问题,因为 `out_cache_loc` 在 pad 时被清零,pad 位置会翻译到 SWA sentinel。piecewise runner(piecewise_cuda_graph_runner.py:654-655)本来就正确清零,本修复使主 runner 对齐该行为。
2. dtype 不匹配:`out_cache_loc_swa` 按 `torch.int64` 分配,但 `SWAKVPool.translate_loc_from_full_to_swa`(swa_memory_pool.py:170)与 DSV4 等价实现(deepseek_v4_memory_pool.py:482)返回 `torch.int32`,拷贝时隐式拓宽 int32→int64,捕获的 graph 把 int64 索引张量交给 `SWAKVPool.set_kv_buffer`。该问题在 #23552 review 中被指出但搁置,本 PR 顺手修复。

## 修复方案
1. `cuda_graph_runner.py`:`out_cache_loc_swa` 分配 dtype 由 int64 改 int32;`populate_from_forward_batch` 在 pad 时整 buffer 清零:

```diff
         if bs != raw_bs:
             self.seq_lens.fill_(seq_len_fill_value)
             self.out_cache_loc.zero_()
+            # Padded SWA indices left over from a previous replay would point
+            # into real SWA slots, so set_kv_buffer on padded tokens would
+            # corrupt active requests' KV. Zero the whole buffer so padded
+            # positions map to the sentinel slot (matches piecewise runner).
+            if self.out_cache_loc_swa is not None:
+                self.out_cache_loc_swa.zero_()
```

2. `piecewise_cuda_graph_runner.py`:同处分配 dtype 同步改为 int32。

## 影响与回归风险
- 影响窗口:回归由 PR #23552 引入(其新增 `out_cache_loc_swa` 预计算 buffer),即 #23552 合并至本修复之间的提交受影响。
- 回归面:hybrid-SWA 模型(Gemma、MiMoV2、DeepSeek V4 等)的 decode CUDA graph 主 runner 与 piecewise runner 的 buffer dtype/清零逻辑;非 hybrid-SWA 模型该 buffer 为 None,不受影响。修复后精度全面回升(见上表),正文未附性能数据。

## 关联
- [PR #23552](https://github.com/sgl-project/sglang/pull/23552)(引入 `out_cache_loc_swa` buffer 的 PR,回归来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
