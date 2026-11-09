# PR #17355: [Diffusion] Fix the LoRA weights mismatch caused by weights packing

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17355](https://github.com/sgl-project/sglang/pull/17355) |
| 作者 | linfann |
| 合并日期 | 2026-01-21 |
| labels | lora, run-ci, diffusion |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
PR #15201 把 w1/w3 打包成 w13 后 LoRA 参数名匹配不上、权重堆叠方式错误,生成结果异常;补 LoRA 映射并用 stack 融合。

## 问题现象
Diffusion 模型(Z-Image-Turbo)加载 LoRA adapter(如 `reverentelusarca/elusarca-anime-style-lora-z-image-turbo`)生成时输出异常:正文给出三张对比图——PR #15201 之前的版本正常、「version now (before this PR)」异常、本 PR 修复后恢复正常。复现命令为 `sglang generate --model-path=Tongyi-MAI/Z-Image-Turbo ... --lora-path=... --enable-torch-compile`。

## 根因
正文披露:PR #15201 将 gate/up proj 打包,fuse `feed_forward.w1 (n, m)` 与 `feed_forward.w3 (n, m)` 为 `feed_forward.w13 (n * 2, m)`;由于参数名改变,LoRA 参数无法匹配,LoRA 参数也需要相应地 fuse 与调整(diff 另显示:合并时用 `torch.cat(dim=1)` 与打包后的 3 维堆叠权重不兼容,需改为 `torch.stack`)。

## 修复方案
1. `python/sglang/multimodal_gen/configs/models/dits/zimage.py`:在 `param_names_mapping` 中为 `w1`/`w3` 的 `lora_A`/`lora_B` 增加到 `w13` 的映射(merge_index 分别为 0/1);
2. `python/sglang/multimodal_gen/runtime/pipelines_core/lora_pipeline.py` `load_lora_adapter`:对 `merge_index is not None` 的所有 LoRA 参数(原来仅处理 `lora_B`)收集合并;合并时由 `torch.cat(sorted_tensors, dim=1)` 改为 `torch.stack(sorted_tensors, dim=0)`(注释:与 TP 兼容);
3. `python/sglang/multimodal_gen/runtime/lora/linear.py`:`forward` 与 `_merge_lora_into_data` 中,当 delta 为 3 维(堆叠形式)时 `reshape(-1, delta.shape[-1])` 后再与基层输出相加。

```diff
-            if merge_index is not None and "lora_B" in name:
+            if merge_index is not None:
                 to_merge_params[target_name][merge_index] = weight
-                    weight = torch.cat(sorted_tensors, dim=1)
+                    # Use stack instead of cat because it needs to be compatible with TP.
+                    weight = torch.stack(sorted_tensors, dim=0)
```

## 影响与回归风险
- 影响窗口:自 PR #15201 引入 w1/w3 打包起、至本修复合并前;影响使用 LoRA 的 multimodal_gen(Diffusion)路径。
- 回归面:multimodal_gen 的 LoRA 加载与合并链路——zimage 的参数名映射、lora_pipeline 的多参数融合(stack 需所有 merge 分量齐全,cat 改 stack 改变了张量形状语义,由 linear.py 侧的 reshape 配套消化)、LoRA linear 的 forward 与预合并路径;TP 并行下的 LoRA 合并行为也随 stack 改变。

## 关联
- https://github.com/sgl-project/sglang/pull/15201 (引入打包的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
