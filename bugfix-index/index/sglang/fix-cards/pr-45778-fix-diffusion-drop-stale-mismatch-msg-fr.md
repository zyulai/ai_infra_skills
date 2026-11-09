# PR #45778: [Fix] diffusion: Drop stale mismatch_msg from FLUX 3 video VAE gate call

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#45778](https://github.com/sgl-project/sglang/pull/45778) |
| 作者 | alexnails |
| 合并日期 | 2026-10-22 |
| labels | diffusion |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

FLUX 3 video VAE 调用 bit-exact fusion gate 时残留已删除的 `mismatch_msg` 关键字致 TypeError;删除该参数修复。

## 问题现象

FLUX 3 video VAE 中第一处 fused QKV norm+RoPE 调用抛出:

```
TypeError: unexpected keyword argument 'mismatch_msg'
```

具体地,`test/registered/kernels/ops/diffusion/test_flux3_video_qknorm_rope.py` 的 `test_natten_prepare_qkv_matches_eager_and_reuses_rope_cache` 在只有 #45775 时仍以该 TypeError 失败;本修复后同文件 4 个测试全部通过(H200 验证)。

## 根因

按 PR 正文:#41894 调用 `BitExactFusionGate.accept_or_fallback(..., mismatch_msg=...)`,但 #44690 已从该 gate 及其他所有调用点移除了 `mismatch_msg` 关键字。FLUX 3 video VAE 的调用点被遗漏,仍传该关键字,触发 `TypeError`。这是 #45775 修复后的遗留断点("This is the remaining break after #45775")。

## 修复方案

改 `python/sglang/multimodal_gen/runtime/models/vaes/flux3_video_vae.py` 的 `_prepare_qkv`,从 `accept_or_fallback` 调用中删除过期的 `mismatch_msg` 参数:

```diff
                         (ref_q, ref_k, ref_v),
                         equal=tensors_equal,
                         logger=logger,
-                        mismatch_msg=(
-                            "FLUX 3 video VAE QKV norm+RoPE is not bit-exact "
-                            "here; using eager"
-                        ),
                     )
```

仅删除关键字参数,fallback 逻辑本身不变。

## 影响与回归风险

- 影响窗口:#44690 移除关键字之后、本修复之前;#45775 修了其余断点,此处是最后一个。
- 回归面:仅 FLUX 3 video VAE 的 `_prepare_qkv` 一处调用签名,与其他调用点对齐;gate 的判定与 eager fallback 行为不变,不影响其他 diffusion 模型路径。验证为 H200 上目标测试文件 4/4 通过。

## 关联

- #41894(引入 mismatch_msg 调用)
- #44690(移除该关键字)
- #45775(先行修复,本次为其遗漏点)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
