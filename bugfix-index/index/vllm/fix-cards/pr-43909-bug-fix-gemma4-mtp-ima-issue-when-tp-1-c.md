# PR #43909: [Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43909](https://github.com/vllm-project/vllm/pull/43909) |
| 作者 | yewentao256 |
| 合并日期 | 2026-05-30 |
| labels | bug, nvidia, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
TP>1 时 gemma4 MTP 的临时 lm_head 权重被释放、CUDA graph replay 撞 IMA,改为缓存稳定引用。

## 问题现象
按正文复现命令(`vllm bench latency --model google/gemma-4-E2B-it --tensor-parallel-size 2 --input-len 512 --output-len 128 --batch-size 16 ... --speculative-config '{"method": "mtp", "model": "gg-hf-am/gemma-4-E2B-it-assistant", "num_speculative_tokens": 1, "use_local_argmax_reduction": false}'`,并 `CUDA_LAUNCH_BLOCKING=1`)会命中:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

出错栈位于 `vllm/v1/spec_decode/gemma4.py` 的 `_greedy_sample` 中 `self._centroids_graphs[size].replay()`。即 TP>1 且启用 gemma4 MTP 投机解码时,CUDA graph replay 阶段触发非法内存访问崩溃。

## 根因
正文披露:`_get_full_lm_head_weight` 返回的临时变量 `lm_head_weight`(TP>1 时对切分权重做 gather/slice 后产生)会被释放,在 CUDA graph replay 时造成问题("temple variable `lm_head_weight` will be released and cause trouble during replay")。机制上,该函数每次调用都新建一个派生张量,graph replay 期望其底层存储持续有效,张量被回收后 replay 便访问到已释放显存(由 diff 推断)。

## 修复方案
修改 `vllm/model_executor/models/gemma4_mtp.py`:
- `__init__` 增加 `self._stable_full_lm_head_weight: torch.Tensor | None = None` 缓存位;
- `_get_full_lm_head_weight` 命中缓存直接返回;TP>1 时对截断后的权重调用 `.contiguous()` 并存入缓存:

```diff
+        lm_head_weight = lm_head_weight[: self.masked_embedding.vocab_size]
+        if tp_size > 1:
+            lm_head_weight = lm_head_weight.contiguous()
+            self._stable_full_lm_head_weight = lm_head_weight
         return lm_head_weight
```

- `load_weights` 开头将 `self._stable_full_lm_head_weight = None`,权重重新加载时使缓存失效。修复后正文基准显示 Avg latency 0.6436000553978374 秒,bench 正常跑完。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅 `gemma4_mtp.py` 的 lm_head 权重获取与加载路径。影响所有 TP>1 的 gemma4 MTP 使用场景;`.contiguous()` 会额外驻留一份完整 lm_head 副本(显存占用略增);`load_weights` 里清缓存保证权重热更新后不引用旧张量。TP=1 路径不写缓存,行为不变。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
