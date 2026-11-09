# PR #59126: [Bugfix][Multimodal] Fix GLM-5.3-Flash vision tower crashes on image input

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59126](https://github.com/vllm-project/vllm/pull/59126) |
| 作者 | gaby |
| 合并日期 | 2026-10-02 |
| labels | bug, glm, ready |
| 热度 | 👍 0 · 💬 13 |
| 分类 | Model-specific Fast Path |
| bug 类型 | crash |

## 一句话结论
GLM-5.3-Flash 视觉塔两类崩溃:pageable 内存非阻塞拷贝触发 GPU sync check、细长图片越界 rope 表;改用 async_tensor_h2d 并按 max_position_embeddings 扩表修复。

## 问题现象
两条独立路径使图片请求打挂引擎:
1. 开启 `VLLM_GPU_SYNC_CHECK=error` 时,任何一张图都会触发报错 "Implicit GPU<->CPU sync detected - use a pinned, densely laid out CPU tensor, or wrap with gpu_sync_allowed()" 并致引擎退出。
2. 细长(slim)图片使视觉 rope 表越界:经 transformers `Glm5NextImageProcessor` 的 `smart_resize` 只按 token 预算约束、无长宽比上限,细长图的网格边长会超过硬编码 `max_position=8192`。例:200000×20 px 图片按 checkpoint 的 processor_config.json 以 7143 tokens 通过,`image_grid_thw = [1, 2, 14286]`,`cos[pos_ids]` 越界——CPU 上是 `IndexError`,GPU 上是 device-side assert。两 bug 均与量化/KV cache dtype 无关。

## 根因
正文披露:
1. `rot_pos_emb` 在 host 构建 `pos_ids` 后从 pageable 内存做 `.to(device, non_blocking=True)`,eager `forward` 对 `cu_seqlens` 同样处理;pageable 内存的非阻塞拷贝可能阻塞 host,触发 sync check。
2. 自 #57387 起图片改走 transformers 的 image processor(无长宽比约束),而 rope 表仍按硬编码 8192 建,细长图越界读表。

## 修复方案
改 `vllm/models/glm5next/common/multimodal.py`:
- rope 表按配置取尺寸:`max_position=8192` → `max_position=text_config.max_position_embeddings`(GLM-5.3-Flash 为 1,048,576 行,每 GPU 约 64 MiB bf16);
- 两处 host→device 拷贝改走 pinned memory 暂存:`pos_ids.to(cos.device, non_blocking=True)` 与 `cu_seqlens.to(self.device, non_blocking=True)` 均改为 `async_tensor_h2d(...)`,与 `Glm4vVisionTransformer` 处理位置表的方式一致。

对旧表能容纳的所有网格,视觉塔输出与修复前 bit-identical。作者注明调查、代码、测试与描述由 Claude Code 协助完成。

## 影响与回归风险
- 影响窗口:#57387 合并(image processor 切换)之后、本 PR 之前;正文以 #57387 为参照,未给出具体版本号。
- 回归面:仅 GLM-5.3-Flash(`Glm5NextForConditionalGeneration`)视觉塔:rope 表内存增大至约 64 MiB/GPU,超出 8192 的网格从崩溃变为可用;两处拷贝改 pinned 内存,old-vs-new 验证显示既有输入输出 bit-identical。GPU 路径的 sync-check 测试在撰写时尚未实机运行(正文自述)。

## 关联
- vllm-project/vllm#57387(引入 transformers image processor)
- vllm-project/vllm#55389(输入归一化,非重复修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
