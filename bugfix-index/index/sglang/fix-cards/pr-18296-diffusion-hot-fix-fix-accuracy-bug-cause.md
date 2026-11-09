# PR #18296: [diffusion][hot fix] fix accuracy bug caused by PR 14717

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18296](https://github.com/sgl-project/sglang/pull/18296) |
| 作者 | yingluosanqian |
| 合并日期 | 2026-02-05 |
| labels | diffusion, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
PR 14717 改了 RMSNorm weight 默认 dtype,导致部分扩散模型生成错误图像;恢复默认 dtype,并顺带修复 cutedsl 归约的同步问题。

## 问题现象
[PR 14717](https://github.com/sgl-project/sglang/pull/14717) 引入精度问题,导致部分模型生成不正确的图像;正文贴出修复前 `flux_image_t2i` 与 `flux_2_klein_image_t2` 的错误输出截图(图像明显异常),修复后图像质量恢复正常。另外正文报告一个 cutedsl 同步问题(明确说明「模型精度不是由这个问题导致的」),表现为单测失败:`python/sglang/jit_kernel/tests/test_fused_norm_scale_shift.py::TestFusedScaleResidualNormScaleShift::test_dtype_1[dtype1-layer]` 断言 `Tensor-likes are not close!`,353894400 个元素中 75 个不匹配(0.0%),最大绝对差 0.59765625(允许 0.05);修复后 148 passed。

## 根因
正文披露:「This issue was caused by a change to the default dtype of the RMSNorm weight」,即 PR 14717 改动了 RMSNorm weight 的默认 dtype,导致上述图像生成精度问题;恢复后图像质量正常。cutedsl 部分正文仅称「a cutedsl sync issue」;由 diff 推断:`cta_reduce_sum` 中各 warp 的部分和与最终结果共用 `acc[0]` 槽位,多个 warp 的 lane 0 并发写 `acc[0]` 与其他 warp 读取 `acc[lane_id]` 存在竞争,故为最终结果单开一个槽位。

## 修复方案
- `python/sglang/multimodal_gen/runtime/layers/layernorm.py`:RMSNorm weight 恢复默认 dtype,不再显式传入:
```diff
-        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
+        self.weight = nn.Parameter(torch.ones(hidden_size))
```
- `python/sglang/jit_kernel/diffusion/cutedsl/common/reduce.py` 的 `cta_reduce_sum`:共享内存分配从 `num_warps` 个 Float32 槽改为 `num_warps + 1` 个,最终结果写入/读取 `acc[num_warps]` 而非 `acc[0]`,与各 warp 部分和槽位分离。
- `python/sglang/jit_kernel/tests/test_fused_norm_scale_shift.py`:调整 `SHAPES` 顺序,把两个多 head 形状 `(1, 1024, 8, 3072)`、`(4, 512, 16, 3072)` 移到列表末尾。

## 影响与回归风险
- 影响窗口:自 PR 14717 合并起至本 PR 之前(正文称问题由 PR 14717 引入)。
- 回归面:所有走 `multimodal_gen` RMSNorm 的扩散模型,weight dtype 从显式 dtype 恢复为参数默认(影响显存与数值路径);`cta_reduce_sum` 被扩散 cutedsl 内核复用,槽位布局变化影响其所有调用方;测试文件顺序调整不影响运行时。

## 关联
- [PR 14717](https://github.com/sgl-project/sglang/pull/14717)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
