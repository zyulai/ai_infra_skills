# PR #14383: [AMD] fix the regression issue for DeepseekV3 on MI300

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14383](https://github.com/sgl-project/sglang/pull/14383) |
| 作者 | yctseng0211 |
| 合并日期 | 2025-12-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论

#13960 把 MI300 上 `if _use_aiter` 权重预shuffle段改成 `elif` 后被跳过,DeepSeek-V3 精度归零;在 fp8.py 中补回该段。

## 问题现象

MI300 上 DeepSeek-V3-0324(开 EAGLE 投机解码,`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`,TP8)出现精度回归。用 `TestDeepseekV3MTP.test_a_gsm8k` 测试:Accuracy 0.000、Invalid 0.690、Latency 162.072 s、Output throughput 631.816 token/s、avg_spec_accept_length=1.0063940886699507。启动命令为 `python3 -m sglang.launch_server --model-path /models/deepseek-ai/DeepSeek-V3-0324 --tp 8 --trust-remote-code --speculative-algorithm EAGLE ... --device cuda`。

## 根因

正文披露:精度回归的根因是 PR-13960 的提交。在 PR-13960 之前,MI300 上权重处理代码会同时命中两个条件分支 `if _is_fp8_fnuz` 与 `if _use_aiter`;PR-13960 把第二个条件 `if _use_aiter` 改成了 `elif _use_aiter`,导致 `_use_aiter` 下的代码段(权重预 shuffle 等)被跳过,从而在 MI300 上引发精度问题。

## 修复方案

改 `python/sglang/srt/layers/quantization/fp8.py` 的 `process_weights_after_loading`:在 `_is_fp8_fnuz` 分支内为 MI300 补回 `_use_aiter` 时的权重预 shuffle 逻辑,使 MI300 上两个分支的逻辑都能执行:

```diff
                 layer.w2_input_scale = None
+                if _use_aiter:
+                    # add this section for MI300
+                    # Pre-shuffle weights
+                    layer.w13_weight.data = shuffle_weight(
+                        layer.w13_weight.contiguous(), (16, 16)
+                    )
+                    layer.w2_weight.data = shuffle_weight(
+                        layer.w2_weight.contiguous(), (16, 16)
+                    )
             elif _use_aiter:
```

修复后验证(DeepSeek-V3-0324,`test_deepseek_v3_mtp.py`):MI300 Accuracy 0.950 / Invalid 0.000 / avg_spec_accept_length=3.00316551100;MI355 Accuracy 0.955 / Invalid 0.000。另测 MI300 + DeepSeek-R1-0528(`SGLANG_USE_AITER=1`、TP8、gsm8k 2000 题):Accuracy 0.948 / Invalid 0.000。

## 影响与回归风险

- 影响窗口:含 PR-13960 改动之后的版本受影响(正文指明回归由 PR-13960 引入;未给出具体版本号)。
- 回归面:改动位于 fp8 量化 `process_weights_after_loading` 的 `_is_fp8_fnuz` 分支内,仅在该分支且 `_use_aiter` 为真(AMD MI300 等场景)时新增权重 shuffle;非 AMD 或非 fnuz 路径不受影响。注意与外层已有的 `elif _use_aiter` 分支形成两条独立 shuffle 路径。

## 关联

- PR-13960(正文指明为回归引入者,以链接形式引用其 commit:7ab548ef641b41ee36da82b67b68761987b39b7c)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
