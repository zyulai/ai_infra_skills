# PR #19964: [diffusion] fix TeaCache silently fails with --enable-teacache

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19964](https://github.com/sgl-project/sglang/pull/19964) |
| 作者 | eitanturok |
| 合并日期 | 2026-03-07 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Req 上显式的 teacache_params=None 挡住了 __getattr__ 委托,TeaCache 被静默禁用;删除该字段让查找代理到 sampling_params。

## 问题现象
指定 `--enable-teacache` 后 TeaCache 被静默禁用(silently disabled):不报错,但缓存逻辑完全不生效。正文用 Wan2.1-T2V-1.3B 在 A100 上验证:main 分支上开 teacache 的生成耗时(150.74 秒)与不开的 baseline(150.62 秒)几乎相同,说明加速完全没生效;修复后的分支同样命令生成 86.46 秒,约 1.7x 加速(跳过部分步骤,视频质量略降属预期,与 main 上 baseline 输出质量一致)。

## 根因
正文明确披露:`Req.teacache_params` 默认值为 `None`,阻止了属性查找向 `sampling_params` 委托。具体链条:

1. 模型专属配置(如 `WanT2V_1_3B_SamplingParams`)会正确填充 `sampling_params.teacache_params`;
2. 但 `Req` 对象把自身的 `teacache_params` 初始化为 `None`;
3. `Req` 通过 `__getattr__` 做委托,只有当属性在 `Req` 上不存在时才会查 `sampling_params`;
4. 由于 `Req.teacache_params` 存在(值为 `None`),委托永远不会发生;`cache/teacache.py` 中只要 `forward_batch.teacache_params` 为 `None` 就绕过缓存,TeaCache 于是被静默关闭。

## 修复方案
从 `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py` 的 `Req` 类中删除显式的 `teacache_params: TeaCacheParams | WanTeaCacheParams | None = None` 字段(同时删除不再需要的 `TeaCacheParams`/`WanTeaCacheParams` import)。删除后访问 `teacache_params` 会命中 `__getattr__`,正确代理到 `sampling_params.teacache_params`,模型专属配置得以生效。

```diff
-    # TeaCache parameters
-    teacache_params: TeaCacheParams | WanTeaCacheParams | None = None
-
     # STA parameters
     STA_param: list | None = None
```

diff 同时更新了 `multimodal_gen/test/server/perf_baselines.json` 中 `wan2_1_t2v_1.3b_teacache_enabled` 的基线:e2e 从 18382.19 ms 降到 6497.84 ms,denoise 步耗时从每步约 247-250 ms 变为大量 3-50 ms 的跳步形态(avg 91.85 ms),反映 teacache 真正生效后的性能形态。

## 影响与回归风险
- 回归面:影响 multimodal_gen 扩散流水线中所有访问 `Req.teacache_params` 的路径——删除字段后,若某处代码在 `Req` 实例上直接赋值/读取该字段而未经 `sampling_params`,行为会改变;TeaCache 从「总是关闭」变为「按 sampling_params 配置生效」,开启 teacache 的生成输出会与旧版本不同(步数被跳过)。该修复是 #19957 的前置条件。

## 关联
- https://github.com/sgl-project/sglang/pull/19957 (本 PR 为其前置条件)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
