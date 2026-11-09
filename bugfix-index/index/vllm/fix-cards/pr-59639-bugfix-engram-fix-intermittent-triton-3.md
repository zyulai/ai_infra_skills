# PR #59639: [Bugfix][Engram] Fix intermittent Triton 3.8 crash in the lookup kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59639](https://github.com/vllm-project/vllm/pull/59639) |
| 作者 | djramic |
| 合并日期 | 2026-10-01 |
| labels | DSv4.1, bug, deepseek, ready |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
运行期对 SORTED 分支会间歇性触发 Triton 3.8 RemoveLayoutConversions 崩溃;在 Triton 3.8+ROCm 上把 SORTED 变为 constexpr 编译期消解。

## 问题现象
AMD CI(MI355 Miscellaneous Kernels,build 13954)中 `test_engram` 间歇性失败;`_engram_lookup_kernel` 在运行期按 `SORTED` 分支时,Triton 3.8 的 `RemoveLayoutConversions` pass 间歇性崩溃(`PassManager::run failed`)。崩溃可命中任何 lookup,包括 unsorted 的调用。

## 根因
正文披露:运行期对 `SORTED` 分支(SORTED 由 #59327 引入)会间歇性触发 Triton 3.8 `RemoveLayoutConversions` pass 的崩溃;上游根因是悬空的 `rematMapping` keys,已由 [triton-lang/triton#10706](https://github.com/triton-lang/triton/pull/10706) 修复,但该修复不在 Triton `release/3.8.x` 中。

## 修复方案
改 `vllm/models/deepseek_v41/common/engram.py` 的 `_engram_lookup_kernel`:当 `triton.__version__` 以 3.8 开头且为 ROCm 平台时,把 `SORTED` 从运行期参数改为 `tl.constexpr`,使分支在编译期消解(绕开有 bug 的 pass);该变量同时从 `do_not_specialize` 列表移除。diff 注释留 TODO:Triton 合入 #10706 后应移除该检查。

```diff
+_SORTED_IS_CONSTEXPR = tl.constexpr(
+    triton.__version__.startswith("3.8") and current_platform.is_rocm()
+)
...
-        "SORTED",
+        *(() if _SORTED_IS_CONSTEXPR else ("SORTED",)),
```

验证(MI355X,`pytest tests/kernels/test_engram.py -k head_shards_reconstruct`,每配置 20 次):main + Triton 3.8.0 失败 14/20;main + 本 PR 失败 0/20;main + Triton #10706 修复 0/20。正文注明本 PR 开发中使用了 AI 辅助,提交者人工复核全部改动并运行上述测试。

## 影响与回归风险
- 影响窗口:Triton 3.8.x + ROCm(尤其 MI355X)环境;正文未给出 vLLM 版本范围。
- 回归面:仅 deepseek_v41 engram lookup kernel 的编译期元参数;Triton 非 3.8 或非 ROCm 时 `_SORTED_IS_CONSTEXPR` 为 False,行为与之前完全一致;Triton 3.8+ROCm 上 SORTED 变为 constexpr,unsorted/sorted 两种调用分别单独编译。

## 关联
- 引入 SORTED 的 [PR #59327](https://github.com/vllm-project/vllm/pull/59327)(diff 注释未注链接,正文未单独列出,由卡片作者按 pack 信息关联)
- 上游修复:[triton-lang/triton#10706](https://github.com/triton-lang/triton/pull/10706)
- [AMD CI build 13954](https://buildkite.com/vllm/amd-ci/builds/13954#01a0f6b2-7be7-4b87-9033-677cce28c6dd)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
