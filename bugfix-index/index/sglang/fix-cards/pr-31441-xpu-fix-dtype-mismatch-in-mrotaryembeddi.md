# PR #31441: [XPU] Fix dtype mismatch in MRotaryEmbedding.forward_xpu by calling _…

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31441](https://github.com/sgl-project/sglang/pull/31441) |
| 作者 | jiayisunx |
| 合并日期 | 2026-07-17 |
| labels | intel, xpu, run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
XPU 上多模态 MRoPE 的 forward_xpu 以 float32 cos_sin_cache 配 bfloat16 Q/K 触发 RuntimeError,现内核调用前先对齐 cache dtype。

## 问题现象
XPU 上的多模态 MRoPE 回归:运行到 `MRotaryEmbedding.forward_xpu` 时报错 `RuntimeError: expected scalar type BFloat16 but found Float`(float32 的 cos_sin_cache 与 bfloat16 的 Q/K 同时传入内核)。正文未描述更大范围的影响面与具体触发模型/负载。

## 根因
正文披露:`forward_xpu` 可能把 float32 的 `cos_sin_cache` 与 bfloat16 的 Q/K 一并传给内核, dtype 不匹配触发 RuntimeError。即 XPU 路径缺少调用内核前的 cos_sin_cache dtype 对齐处理(其他路径已有该处理,由 diff 推断:修复即在 `forward_xpu` 补调与 CUDA 等路径同名的 `_match_cos_sin_cache_dtype`)。

## 修复方案
改 `python/sglang/srt/layers/rotary_embedding/mrope.py` 的 `forward_xpu`:在进入 multimodal_rotary_embedding 内核调用前,先用 query 的 dtype 对齐 cos_sin_cache:

```diff
     ) -> Tuple[torch.Tensor, torch.Tensor]:
         assert positions.ndim in (1, 2)
+        self._match_cos_sin_cache_dtype(query)
         if positions.ndim == 2 and self.mrope_section:
             multimodal_rotary_embedding(
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;仅称其为 XPU 回归(regression)。
- 回归面:`mrope.py` 的 `forward_xpu` 路径(XPU 设备上的多模态 MRoPE 模型)。新增的 dtype 对齐为幂等操作,cos_sin_cache 已是目标 dtype 时应无行为变化;`_match_cos_sin_cache_dtype` 为既有方法,非 CUDA 路径行为不变。正文未提供 accuracy/benchmark 数据;CI:Base 红、Extra 绿(原帖标注)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
