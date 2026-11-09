# PR #14585: [Glm46v] Bug fix for accuracy drop and unable to launch server

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14585](https://github.com/sgl-project/sglang/pull/14585) |
| 作者 | byjiang1996 |
| 合并日期 | 2025-12-08 |
| labels | dependencies, documentation, high priority, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

修复 GLM-4.6V 精度下降与无法启动:按条件自动禁用 shared experts fusion,修正视觉侧 qkv bias 与视频 grid 处理。

## 问题现象

GLM-4.6V 存在 accuracy drop,并且无法启动服务器(标题与正文,修复 issue #14582)。正文说明当前先以禁用 shared expert fusion 特性作为 workaround 绕过启动问题。正文附精度测试截图,未提供具体数字文本。

## 根因

正文披露:shared experts fusion 存在问题导致服务器无法启动,原文「Currently i just disabled shared expert fusion feature to bypass it as workaround」。accuracy drop 的具体根因正文未披露;由 diff 可见视觉侧两处改动暗示的机制:视觉注意力原本固定 `proj_bias=True`,改为 `proj_bias=False` 并新增 `qkv_bias=attn_qkv_bias`(按 `vision_config.attention_bias` 传入),修正注意力偏置与权重不匹配;`get_video_feature` 将 `video_grid_thw` 展平为逐帧 `[1,h,w]` 再传给视觉模型(由 diff 推断)。

## 修复方案

- `python/sglang/srt/models/glm4v.py`:视觉 block 引入 `attn_qkv_bias` 参数,`proj_bias=False` + `qkv_bias=attn_qkv_bias`;`get_video_feature` 把 `video_grid_thw` 按 `(t,h,w)` 展平为逐帧行再拼接。
- `python/sglang/srt/models/glm4v_moe.py`:新增 `determine_num_fused_shared_experts`,在无 shared experts 配置、非 CUDA、SM<80、EP>1、DeepEP 后端等情况自动禁用 fusion 并打印原因;lm_head 支持 PP 非末 ranks 与 `tie_word_embeddings`;`load_weights` 按 `n_routed_experts + num_fused_shared_experts` 处理并重命名 shared expert 权重。
- 其他:`qwen3_vl.py`/`qwen3_omni.py` 移除 `rope_config_validation` 调用;pyproject 将 torchcodec 0.7.0→0.8.0;新增 `Glm4MoeThinkingBudgetLogitProcessor`;`Glm4vForConditionalGeneration` 加入 fa3 默认架构列表;新增 GLM-4.5/4.6 与 GLMV 文档。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:GLM-4.5V/4.6V 的视觉权重加载、shared experts fusion 决策、PP/DP encoder 路径;qwen3_vl/qwen3_omni 配置加载不再做 rope 校验;依赖 torchcodec 的环境版本要求变化。

## 关联

- https://github.com/sgl-project/sglang/pull/14568
- https://github.com/sgl-project/sglang/issues/14582
- https://github.com/sgl-project/sglang/pull/14584

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
