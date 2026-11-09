# PR #29042: [NPU] Fix the DeepSeek-V2-Coder model accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29042](https://github.com/sgl-project/sglang/pull/29042) |
| 作者 | zhsurpass |
| 合并日期 | 2026-06-25 |
| labels | deepseek, run-ci |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
NPU 上 MoE topk 的 norm_type 硬编码为 sigmoid,DeepSeek-V2-Coder(softmax 评分)路由打分错误;按 scoring_func 动态判断修复。

## 问题现象
华为 Ascend NPU 上运行 DeepSeek-V2-Coder(DeepSeek-Coder-V2-Lite-Instruct)模型出现精度问题。正文 Motivation 仅一句「Fix the DeepSeek-V2-Coder model accuracy issue」,未给出量化前后对比数字,仅附精度测试截图(server 配置:NPU 设备、ascend attention backend、TP2、moe-dense-tp-size 1、HCCL_BUFFSIZE=1024)。

## 根因
据正文修改说明披露:NPU 的 `fused_topk_npu()` 此前未根据 scoring 配置决定归一化类型,diff 显示原代码硬编码 `norm_type=1`(注释:1 为 sigmoid、0 为 softmax);同时 `DeepseekV2MoE.__init__` 构造 topk 配置时没有把 `config.scoring_func` 传入 `topk_kwargs`。DeepSeek-V2-Coder 使用 softmax 评分函数,于是路由 topk 被按 sigmoid 归一化计算,专家打分错误,精度受损。

## 修复方案
1. `python/sglang/srt/hardware_backend/npu/moe/topk.py` `fused_topk_npu()`:`norm_type` 由硬编码 `1` 改为 `(0 if topk_config.scoring_func == "softmax" else 1)`,按 `topk_config.scoring_func` 动态选择 softmax/sigmoid。
2. `python/sglang/srt/models/deepseek_v2.py` `DeepseekV2MoE::__init__`:向 `topk_kwargs` 传入 `scoring_func=config.scoring_func`,使 `self.topk` 按构造时的取值正确确定评分函数。
3. `python/sglang/srt/models/llada2.py`:同步为 topk 配置传入 `scoring_func=self.score_function`(正文修改说明未单独提及,属 diff 附带改动)。

```diff
-            norm_type=1,  # 1 for sigmoid, 0 for softmax
+            # 1 for sigmoid, 0 for softmax
+            norm_type=(0 if topk_config.scoring_func == "softmax" else 1),
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:NPU 上所有使用 `fused_topk_npu` 的 MoE 模型的路由归一化逻辑——此前一律 sigmoid,现在 softmax 模型改走 softmax、其余仍 sigmoid,行为变化仅限 scoring_func 为 "softmax" 的模型(如 DeepSeek-V2-Coder);DeepseekV2MoE 与 llada2 的 topk_kwargs 新增 scoring_func 字段,若下游 topk 实现不识别该参数需注意兼容性;CUDA/其他硬件后端的 topk 路径不受影响。

## 关联
正文未列出(仅附精度测试截图与 CI 运行记录链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
