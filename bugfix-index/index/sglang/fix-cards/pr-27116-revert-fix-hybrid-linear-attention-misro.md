# PR #27116: Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27116](https://github.com/sgl-project/sglang/pull/27116) |
| 作者 | ch-wan |
| 合并日期 | 2026-06-03 |
| labels | (none) |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
整体回退 #26623 的 hybrid linear attention 层路由修复,`_is_full_attn` 对普通 RadixAttention 恢复"默认 full、靠 `_is_linear_attention` 标记识别线性层"的判定方式。

## 问题现象
正文未描述具体现象,仅说明本 PR 为「Reverts sgl-project/sglang#26623」,即回退标题为"Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"的修复。被回退修复针对的问题是:混合线性注意力模型(Ring-2.5-1T 等)中用普通 RadixAttention 实现的线性层被误路由到 full attention 后端;本 PR 正文未说明回退的具体触发故障或测试表现。

## 根因
正文未披露根因,仅声明回退 #26623。(由 diff 推断) 回退后的判定逻辑是:`_is_full_attn` 中显式 `RadixLinearAttention` 子类判为线性;普通 `RadixAttention` 默认判为 full,除非 attn 模块带有 `_is_linear_attention=True` 标记;同时 `bailing_moe_linear.py` 给自己的 attn 模块打上该标记。与之相对,#26623 引入的做法是把 `_is_full_attn` 的参数放宽为 `Union[RadixAttention, RadixLinearAttention]`,并让"包括普通 RadixAttention 在内的一切非 RadixLinearAttention 层"按 layer id 分类(即不再对普通 RadixAttention 走 `isinstance → full` 的捷径)。回退等于放弃后一种"按 layer id 分类"的方案,恢复"类型 + 属性标记"的判别方案。

## 修复方案
整体 revert #26623,涉及两个文件:
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`:`_is_full_attn` 签名恢复为仅接受 `Optional[RadixAttention]`,新增属性标记检查与普通 RadixAttention 的 full 短路:

```diff
+        if layer is not None and getattr(layer, "_is_linear_attention", False):
+            return False
+        if isinstance(layer, RadixAttention):
+            return True
```

- `python/sglang/srt/models/bailing_moe_linear.py`:为 attn 模块添加路由标记:

```diff
+        self.attn._is_linear_attention = True
```

(附注:Bailing/Ring 用普通 RadixAttention 包装线性层,标记使调度器能将其与 full-attention 层区分,包括默认走 full 路径的 MTP/NEXTN draft 层——此为 diff 中注释所述。)

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `hybrid_linear_attn_backend.py` 与 `bailing_moe_linear.py`。影响所有 hybrid linear attention 模型的层路由:普通 RadixAttention 层若既非 RadixLinearAttention 又无 `_is_linear_attention` 标记,将回到 full backend(#26623 试图修的 Ring-2.5-1T 类模型若无标记,误路由问题会重新出现);Bailing 模型经新标记仍走线性路径。MTP/NEXTN draft 层按注释意图默认 full 路径。

## 关联
- Reverts sgl-project/sglang#26623

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
