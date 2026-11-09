# PR #22998: [BugFix] Fix regression caused by mamba state dtype PR

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22998](https://github.com/vllm-project/vllm/pull/22998) |
| 作者 | tdoublep |
| 合并日期 | 2025-08-15 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
mamba state dtype PR(#22928)改了 MambaCacheManager 构造参数,phi4flash/plamo2 按旧序传参出错;修正两处调用。

## 问题现象
正文未给出具体报错日志。测试命令 `VLLM_ATTENTION_BACKEND=DIFFERENTIAL_FLASH_ATTN vllm serve microsoft/Phi-4-mini-flash-reasoning --trust-remote-code` 在修复前失败,修复后 "Now works fine"。另注:plamo2 在最新 transformers 下仍因无关 issue(#22999)而有问题。

## 根因
正文:修复 issue #22994,并解决 mamba state dtype PR(#22913/#22928,正文链接指向 PR 22928)引入的回归;回归的具体机制正文未披露。从 diff 看,#22928 改变了 `MambaCacheManager` 的构造签名(num_mamba_layers 与 cache shape 前移、末尾新增两个 dtype 参数),而 `phi4flash.py` 与 `plamo2.py` 仍按旧参数顺序把 `lm_head.weight.dtype` 传在第 2 位,造成实参错位(由 diff 推断)。

## 修复方案
`vllm/model_executor/models/phi4flash.py` 与 `vllm/model_executor/models/plamo2.py`:两处 `MambaCacheManager(...)` 调用改为新签名——`num_mamba_layers` 与 `*self._get_mamba_cache_shape()` 前移,`self.lm_head.weight.dtype` 作为两个 dtype 参数在末尾传入。

```diff
             self.mamba_cache = MambaCacheManager(
-                self.vllm_config, self.lm_head.weight.dtype, num_mamba_layers,
-                *self._get_mamba_cache_shape())
+                self.vllm_config,
+                num_mamba_layers,
+                *self._get_mamba_cache_shape(),
+                self.lm_head.weight.dtype,
+                self.lm_head.weight.dtype,
+            )
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围(回归由 #22928 引入)。
- 回归面:仅 phi4flash 与 plamo2 两个模型的 mamba cache 初始化调用点;两个 dtype 实参均传 `lm_head.weight.dtype`,与该文件其他缓存 dtype 保持一致。

## 关联
- Issue #22994
- PR #22928(正文写作 #22913,链接指向 22928)
- Issue #22999(plamo2 与 transformers 相关的未解决问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
