# PR #6567: Fix accuracy is zero when enabling moe-dense-tp-size as in large scale EP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6567](https://github.com/sgl-project/sglang/pull/6567) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-05-24 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
大规模 EP 部署开启 --moe-dense-tp-size 1 时输出全错;在 ModelRunner 的 server_args 属性字典补上 enable_dp_lm_head 键后输出正确,gsm8k 0.937。

## 问题现象
复现命令(正文原文):`python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --trust-remote-code --dist-init-addr 10.5.55.13:5757 --nnodes 2 --node-rank 0 --tp-size 16 --dp-size 16 --enable-dp-attention --chunked-prefill-size 32768 --max-running-requests 2048 --context-length 4096 --disable-radix-cache --enable-deepep-moe --deepep-mode normal --moe-dense-tp-size 1 --disable-cuda-graph --enable-dp-lm-head`。before:"output wrong things"(对应标题的 accuracy is zero);after:"output correct things";修复后 gsm8k 0.937。正文未给出更多现象细节。

## 根因
正文未披露根因。从 diff 看,改动仅在 `model_runner.py` `ModelRunner.__init__` 构建的一个 server_args 属性字典(含 disable_radix_cache、enable_nan_detection、enable_dp_attention、enable_ep_moe、enable_deepep_moe、deepep_config 等键)中新增 `"enable_dp_lm_head": server_args.enable_dp_lm_head`(由 diff 推断:此前该字典未携带 enable_dp_lm_head,相关消费方感知不到该开关)。

## 修复方案
一行添加:

```diff
                 "enable_dp_attention": server_args.enable_dp_attention,
+                "enable_dp_lm_head": server_args.enable_dp_lm_head,
                 "enable_ep_moe": server_args.enable_ep_moe,
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:同时启用 dp attention + dp lm head(尤其配合 moe-dense-tp-size)的部署路径;仅向该属性字典加键,未启用相关功能的部署不受影响。作者同时建议"maybe we can refactor this part"(正文原话),提示此处可重构。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
