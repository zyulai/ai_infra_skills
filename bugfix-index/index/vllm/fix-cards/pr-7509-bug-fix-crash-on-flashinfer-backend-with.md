# PR #7509: [BUG] fix crash on flashinfer backend with cudagraph disabled, when attention group_size not in [1,2,4,8]

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7509](https://github.com/vllm-project/vllm/pull/7509) |
| 作者 | learninmou |
| 合并日期 | 2024-08-21 |
| labels | ready |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
flashinfer decode 仅支持 GQA group_size ∈ [1,2,4,8],其余需走 tensor cores;旧判断与 cudagraph 禁用组合导致崩溃,改为 group size 不在支持列表即用 tensor cores。

## 问题现象
使用 flashinfer backend 并禁用 cuda graph、加载 attention group_size=6 的模型时,vllm 崩溃(具体报错日志见原 PR 截图,pack 素材中不含截图内容)。正文给出的稳定复现条件:1) 显式设置 `VLLM_ATTENTION_BACKEND="FLASHINFER"`;2) 模型使用 GQA 且 group size 不在 [1,2,4,8];3) cuda graph 被禁用——`enforce_eager=True`,或 decode batch_size 超过 `model_runner.py` 的 `_BATCH_SIZES_TO_CAPTURE[-1]`(当时为 256)。

## 根因
正文披露:作者查阅 flashinfer 源码(decode.py),flashinfer 的 decode 只支持 group_size ∈ [1,2,4,8],否则用户需要 `use_tensor_cores=True` 才能做 decode。vLLM flashinfer backend 中 `use_tensor_cores` 的取值与该约束不匹配,在 cudagraph 禁用时 decode 走非 graph wrapper,满足上述三个条件即崩溃。具体崩溃栈在正文截图中,pack 未包含;新旧判断条件以 diff 为准如实记录:旧代码 `use_tensor_cores = num_qo_heads // num_kv_heads >= 4`,新代码 `use_tensor_cores = (num_qo_heads // num_kv_heads) not in (1, 2, 4, 8)`。

## 修复方案
`vllm/attention/backends/flashinfer.py`:两处(非 graph 的 `_get_decode_wrapper` 与 `graph_capture_get_metadata_for_batch` 的 graph decode wrapper)把 `use_tensor_cores` 判断从 `num_qo_heads // num_kv_heads >= 4` 改为 `(num_qo_heads // num_kv_heads) not in (1, 2, 4, 8)`,使任何 flashinfer decode 不支持的 group size 都改走 tensor cores 路径。

```diff
-            use_tensor_cores = num_qo_heads // num_kv_heads >= 4
+            use_tensor_cores = (num_qo_heads // num_kv_heads) not in \
+                (1, 2, 4, 8)
```

测试侧 `tests/kernels/test_flashinfer.py` 增加 `(6, 1)` 头配置,并在 `BatchDecodeWithPagedKVCacheWrapper` 构造时按同一规则设置 `use_tensor_cores`。

## 影响与回归风险
- 回归面:flashinfer backend 的 decode wrapper 选择逻辑(cudagraph 开/关两条路径);group size 大于 8 且原条件为 True 的模型(如 group size 6)在两条路径下的行为均改为 tensor cores;group size ≤ 4 的常见配置不受影响(条件结果相同)。

## 关联
正文未列出 issue/PR 链接;引用了 flashinfer 源码链接(github.com/flashinfer-ai/flashinfer/blob/main/python/flashinfer/decode.py)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
