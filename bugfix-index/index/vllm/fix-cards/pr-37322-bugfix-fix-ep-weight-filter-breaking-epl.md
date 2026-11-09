# PR #37322: [Bugfix] Fix EP weight filter breaking EPLB and NVFP4 accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37322](https://github.com/vllm-project/vllm/pull/37322) |
| 作者 | elvircrn |
| 合并日期 | 2026-03-18 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
#37136 的 EP 权重过滤破坏 EPLB 冗余槽位与 NVFP4 scale,改为 EPLB 时跳过、只过滤 .weight 张量。

## 问题现象
PR #37136 引入的 EP weight filter(在从磁盘读权重前跳过非本地 expert 张量以省 I/O)带来两类精度故障:1) 开启 EPLB 时灾难性精度损失,DeepSeek-R1 上 gsm8k 从约 0.95 掉到约 0.08;2) 无 EPLB 时 FlashInfer NVFP4 后端出现约 2% 的精度回归。触发条件:EP 部署 + EPLB,或 EP + NVFP4 量化(非 EPLB)。

## 根因
正文披露两处根因:1) EPLB 启用时,冗余物理 expert 槽位(如 DeepSeek-R1 的 256 逻辑 + 32 冗余 = 288 物理)可映射到任意逻辑 expert,而 EP weight filter 按 `model_config.get_num_experts()`(256 逻辑 expert)划分并跳过非本地项;但权重加载器经 `make_expert_params_mapping` 需要全部 256 个逻辑 expert 权重去填充冗余槽位,过滤后冗余槽位未被初始化。2) FlashInfer NVFP4 后端通过对所有 expert 槽位的 `a13_scale.max()` 计算全局激活 scale;filter 跳过了非本地 expert 的 `input_scale` checkpoint 张量,这些槽位停留在 `torch.empty()` 的随机垃圾值上,`max()` 取到垃圾值,得到错误的全局激活 scale。

## 修复方案
两处修复:
- `vllm/model_executor/model_loader/default_loader.py` 的 `_init_ep_weight_filter`:开启 `parallel_config.enable_eplb` 时直接 return,完全跳过该 filter;
- `vllm/model_executor/model_loader/ep_weight_filter.py` 的 `should_skip_weight`:只允许跳过以 `.weight` 结尾的大张量,scale/元数据张量(`input_scale`、`weight_scale`、`weight_scale_2`)始终加载。

```diff
+    # Only skip heavy weight tensors, never scale/metadata tensors.
+    if not weight_name.endswith(".weight"):
+        return False
     return eid not in local_expert_ids
```

## 影响与回归风险
- 影响窗口:自 #37136 引入 EP weight filter 起,至本修复合并止。
- 回归面:EP 场景的权重加载路径。EPLB 场景恢复全量读盘,I/O 节省在 EPLB 下主动放弃;非 EPLB 下 filter 仍对 `.weight` 生效,I/O 节省保留,但 scale 类小张量恢复加载(开销可忽略)。作者已在 DeepSeek-R1-0528-FP4-v2(GB200 NVL72, EP16, 32 冗余 expert)上验证 EPLB 精度恢复,lm_eval gsm8k 从 0.93 回到 0.95。

## 关联
- PR #37136(引入问题的改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
