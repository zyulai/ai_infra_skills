# PR #18213: [Bugfix] Fix model output corruption caused by EPLB rebalance (Eager and CUDA Graph modes)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18213](https://github.com/sgl-project/sglang/pull/18213) |
| 作者 | lviy |
| 合并日期 | 2026-03-18 |
| labels | deepseek, run-ci |
| 热度 | 👍 1 · 💬 7 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论

DeepSeek-V2 系的 `forward_normal` 与双流 forward 在启用 EPLB 时未对 expert_id 做 logical→physical 映射,rebalance 之后 topk 派发到错误物理专家、输出严重损坏;补上 dispatch_info 修复。

## 问题现象

开启 EPLB 推理时出现严重的模型输出损坏(正文附截图),与 issue #15680 类似。复现命令(作者在 H800 x4 上):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path .../moonshotai/Moonlight-16B-A3B-Instruct \
  --trust-remote-code --tp 4 --ep 4 --dp 1 \
  --moe-a2a-backend none \
  --enable-eplb --eplb-rebalance-num-iterations 10 \
  --expert-distribution-recorder-buffer-size 10
```

作者通过手动控制 load rebalancing 的触发时机确认:输出劣化**专门发生在 rebalancing 阶段之后**——rebalance 前输出正常,rebalance 执行一次即损坏。模型可替换为任何基于 deepseek_v2.py 的模型(正文在 H800 x2 的复现用 `--tp 2 --ep 2`、`--eplb-rebalance-num-iterations 50`)。

## 根因

正文披露:通过 logging 进一步调查,根因是 `deepseek_v2.py` 的 `forward_normal` 与 `forward_normal_dual_stream`(正文写作 dual_forward_normal,用于 CUDA Graph)都缺少 expert_id 必需的 logical-to-physical 映射——两处 `self.topk(hidden_states, router_logits)` 调用均未传入 `expert_location_dispatch_info`。启用 EPLB 后专家会被 rebalance 迁移到新的物理位置,而这两条 forward 路径仍按 logical id 派发,token 被路由到错误的物理专家,输出随之损坏。

## 修复方案

在两条 forward 路径中,当 `server_args.enable_eplb` 时构造 dispatch info 并传给 topk:

关键摘录(deepseek_v2.py,`forward_normal_dual_stream` 与 `forward_normal` 各一处):

```diff
+        server_args = get_global_server_args()
+        dispatch_info = (
+            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
+            if server_args.enable_eplb
+            else None
+        )
         ...
-            topk_output = self.topk(hidden_states, router_logits)
+            topk_output = self.topk(
+                hidden_states,
+                router_logits,
+                expert_location_dispatch_info=dispatch_info,
+            )
```

改动仅 `python/sglang/srt/models/deepseek_v2.py` 两个函数。修复后正文在 H800 x2(修复前后截图对比)以及其他配置(H800 x2/4/8)上验证均无问题;另附 `bench_one_batch`(DeepSeek-V2-Lite-Chat)修复前后对比与精度测试截图。正文同时提出:其他 MoE 模型(qwen3_moe.py、mixtral.py)疑似同样的实现模式、可能有同样问题,留给维护者决定是否在后续 PR 中扩展。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,仅 deepseek_v2.py 的 eager forward_normal 与双流(含 CUDA Graph)forward 路径;`enable_eplb=False` 时 dispatch_info 为 None,行为不变。基于 deepseek_v2.py 的其他 MoE 模型(正文点名 qwen3_moe、mixtral 未在本次修复范围)若复用该模式则仍有同类风险。

## 关联

- issue [#15680](https://github.com/sgl-project/sglang/issues/15680)(正文称「This is similar to Issue#15680」)
- 正文 CC @FortPercent @qzweng

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
