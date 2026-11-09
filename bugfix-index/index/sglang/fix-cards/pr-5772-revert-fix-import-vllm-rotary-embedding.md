# PR #5772: Revert "fix: import vllm_rotary_embedding error when head_size not in 64, 128, 256, 512"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5772](https://github.com/sgl-project/sglang/pull/5772) |
| 作者 | merrymercy |
| 合并日期 | 2025-04-27 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

回滚 #5733:该改动导致 CI 中 EAGLE verify 报 CUDA IMA;恢复 rotary_embedding 的模块级导入与调用方式。

## 问题现象

CI 中调度器崩溃。正文给出错误日志(2025-04-27 05:08:35 TP0):EAGLE 投机解码路径 `eagle_utils.py` verify 中 `accept_index_cpu = accept_index.tolist()` 处抛 `RuntimeError: CUDA error: an illegal memory access was encountered`。日志链接为 Actions run 14687872443/job/41218993586。作者给出本地复现方式:`test/srt` 目录下 `python3 test_eagle_infer.py`。

## 根因

正文披露:错误由 #5733("fix: import vllm_rotary_embedding error when head_size not in 64, 128, 256, 512")引入,但未说明其导致 IMA 的机制。(由 diff 推断)#5733 曾把 `vllm_rotary_embedding` 的导入从模块级(非 CUDA 分支)改为在 `__init__` 内按 `not _is_cuda or head_size not in [64, 128, 256, 512]` 条件导入并绑定为实例属性 `self.vllm_rotary_embedding`,forward 调用点相应改为 `self.vllm_rotary_embedding`;本回滚撤销这一改动,恢复模块级导入与直接调用。

## 修复方案

按 revert 提交恢复 `python/sglang/srt/layers/rotary_embedding.py` 三处:

模块顶部恢复 else 分支导入:

```diff
 if _is_cuda:
     from sgl_kernel import apply_rope_with_cos_sin_cache_inplace
+else:
+    from vllm._custom_ops import rotary_embedding as vllm_rotary_embedding
```

`__init__` 中删除 #5733 加入的条件导入块(`if not _is_cuda or self.head_size not in [64, 128, 256, 512]: from vllm._custom_ops import rotary_embedding; self.vllm_rotary_embedding = rotary_embedding`);`forward_cuda` 中调用从 `self.vllm_rotary_embedding(...)` 恢复为模块级 `vllm_rotary_embedding(...)`。

## 影响与回归风险

- 影响窗口:含 #5733 改动的版本受影响(依据正文声明;正文未给出具体版本号)。
- 回归面:改动触及 rotary embedding 层的导入结构与调用路径。恢复后非 CUDA 平台走模块级 vllm 导入;#5733 原本要解决的「head_size 不在 64/128/256/512 时导入报错」问题会重新存在,需要另想方案;EAGLE 投机解码路径的 IMA 随回滚消失。

## 关联

- [#5733](https://github.com/sgl-project/sglang/pull/5733)(正文声明被回滚的 PR)
- [CI 失败日志](https://github.com/sgl-project/sglang/actions/runs/14687872443/job/41218993586#step:4:351)
- @lkm2835(正文 cc 请其排查)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
