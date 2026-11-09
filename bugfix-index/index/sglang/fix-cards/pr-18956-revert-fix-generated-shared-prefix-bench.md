# PR #18956: Revert "Fix generated-shared-prefix bench_serving"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18956](https://github.com/sgl-project/sglang/pull/18956) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-02-18 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论
回滚 #18769:其要修的 np.array 错误实际已由 #18761 修复,属重复改动,恢复 `compute_random_lens` 返回 List[int]。

## 问题现象
正文未描述用户侧或 CI 的具体现象,仅说明:#18769(Fix generated-shared-prefix bench_serving)试图修一个 np.array 错误,但该错误实际上已经在 #18761 中被修复,因此 #18769 属于重复/过时的修复,整体回滚。正文 cc 了 @sufeng-buaa 与 @zminglei。

## 根因
正文披露:#18769 针对的 np.array 错误已由 #18761 修复,该 PR 的改动不再需要。从 diff 看被回退的改动是:`bench_serving.py` 的 `compute_random_lens` 曾由 `np.random.randint(...).tolist()`(返回 `List[int]`)改为直接返回 `np.ndarray`;本回滚恢复 `.tolist()`,使随机长度序列回到 Python list 类型(由 diff 推断)。

## 修复方案
恢复 `python/sglang/bench_serving.py` 的 `compute_random_lens` 签名与返回类型:

```diff
-def compute_random_lens(full_len: int, range_ratio: float, num: int) -> np.ndarray:
+def compute_random_lens(full_len: int, range_ratio: float, num: int) -> List[int]:
     return np.random.randint(
         max(int(full_len * range_ratio), 1),
         full_len + 1,
         size=num,
-    )
+    ).tolist()
```

该函数位于 `sample_custom_requests` 与 `sample_random_requests` 之间,是 bench_serving 生成随机请求长度的路径。

## 影响与回归风险
- 影响窗口:正文未说明(#18769 合并至本回滚之间)。
- 回归面:仅 `python/sglang/bench_serving.py` 的随机长度生成——`compute_random_lens` 的返回类型从 `np.ndarray` 恢复为 `List[int]`,其调用方拿到 list 而非 ndarray;不影响服务端运行时。#18761 中对 np.array 错误的正式修复不受影响。

## 关联
- [#18769](https://github.com/sgl-project/sglang/pull/18769)(被回滚的 PR)
- [#18761](https://github.com/sgl-project/sglang/pull/18761)(对同一 np.array 错误的正式修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
