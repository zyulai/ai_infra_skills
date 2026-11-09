# PR #11437: Revert "fix: fix video input for qwen3-vl"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11437](https://github.com/sgl-project/sglang/pull/11437) |
| 作者 | merrymercy |
| 合并日期 | 2025-10-10 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
回滚 #11361:删除其对 qwen3-vl 视频 rope index 的特殊处理与 fp8 校验,恢复原 get_rope_index 逻辑和视频测试断言。

## 问题现象
正文未描述具体现象,仅一句话说明回滚 #11361(fix: fix video input for qwen3-vl)。没有列出失败的测试、报错日志或复现步骤。

## 根因
正文未披露根因。从 diff 看,回滚内容包括四部分(机制描述均由 diff 推断):(1) 删除 `get_rope_index` 中对 `qwen3_vl` 且 `video_grid_thw` 非空时 `torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)` 并把第 0 维置 1 的特殊处理;(2) 删除 `model_runner.py` 中 `qwen3_vl_moe` fp8 量化下 `moe_intermediate_size // (tp_size // moe_ep_size)` 对 `weight_block_size` 的整除校验;(3) 删除 `TestQwen3VLServer` 端到端测试类;(4) 视频测试断言恢复为大小写敏感的 `"iPod" in video_response`,移除 #11361 引入的 `lower()` 与 `"phone"` 备选——该测试断言被放宽过,暗示 #11361 期间视频问答输出与预期不符(由 diff 推断)。

## 修复方案
- `python/sglang/srt/layers/rotary_embedding.py`:`get_rope_index` 删除 qwen3_vl 视频分支,恢复原 mrope 位置计算
- `python/sglang/srt/model_executor/model_runner.py`:`model_specific_adjustment` 删除 qwen3_vl_moe fp8 整除校验
- `test/srt/test_vision_openai_server_a.py`:删除 `TestQwen3VLServer`
- `test/srt/test_vision_openai_server_common.py`:`test_video_chat_completion` 断言恢复原文

## 影响与回归风险
- 影响窗口:正文未说明(#11361 合并至本回滚之间)。
- 回归面:qwen3-vl 视频输入的 rope index 计算回到 #11361 之前——若 #11361 要修的视频 bug 真实存在,该 bug 会回归(正文未评价);qwen3-vl-fp8 失去启动期整除校验;TestQwen3VLServer 被删,该模型的 OpenAI 视频端到端 CI 覆盖减少。

## 关联
- [#11361](https://github.com/sgl-project/sglang/pull/11361)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
