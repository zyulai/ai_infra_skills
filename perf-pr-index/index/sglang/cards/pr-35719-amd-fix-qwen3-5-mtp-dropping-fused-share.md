# PR #35719: [AMD] Fix Qwen3.5 MTP dropping fused shared-expert weights

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-08-24 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 1 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/35719 |

## 一句话结论

修复 MI355X 上 Qwen3.5 MTP 草稿模型丢失 fused shared-expert 权重的问题,accept rate 0.51→0.65,吞吐回升 12–14%。

## 动机

自 #33889 起,MI355X 上 Qwen3.5 MXFP4 + MTP 投机解码损失约 12% 吞吐。根因是 MTP 草稿模型的 loader 不会把 `mlp.shared_expert.*` 权重装入 fused 槽位:Qwen3.5 MoE 层 fuse shared expert 时不建独立 `shared_expert` 模块,而是把它当作槽位 `num_experts` 的一个 routed expert,`qwen3_5.py` 做了重定向而 `qwen3_5_mtp.py` 没有。目标模型不受影响:该 checkpoint 的 routed experts 为 MXFP4、shared expert 为 bf16,混合精度使 `can_fuse_shared_expert()` 拒绝融合,走非 fused 路径正常加载。MTP 模块整体排除在量化外(全部 bf16),融合合法;#33889 把 `disable_shared_experts_fusion` 从进程级改为 per-runner 后,草稿首次开始融合,随即踩中 loader 缺口。

## 优化点

权重是静默丢失的:expert mapping 只覆盖槽位 0..num_experts-1,`mlp.shared_expert.{gate,up,down}_proj.weight` 匹配不到任何参数,仅留一行被抑制的 warning,每 TP rank 丢 3 个张量,草稿带着未初始化的 shared expert 运行,accept rate 从 0.65 跌到 0.51。修复在 `qwen3_5_mtp.py` 中把 expert mapping 尺寸扩为 `num_experts + num_fused_shared_experts`(融合计数从已构建模块上读取,因为 `num_fused_shared_experts` 挂在 `Qwen3_5MoeForConditionalGeneration` 上而 MTP 包的是 `Qwen3_5ForCausalLM`),并在 mapping 循环前把 `mlp.shared_expert.` 重写为 `mlp.experts.{num_experts}.`。两处均以模型实际融合状态为条件(`_use_aiter` 且融合计数 > 0),非融合配置下完全 no-op。PR 明确未覆盖 fused-checkpoint 变体(单一 `experts.gate_up_proj` 张量),因 MTP loader 的 per-expert 索引无法服务该形态,加未测路径风险更大。

## 关键改动

```text
python/sglang/srt/models/qwen3_5_mtp.py: expert mapping 扩展覆盖 fused shared-expert 槽位,并重写 shared_expert 权重名
```

```python
if (
    _use_aiter
    and num_fused_shared_experts > 0
    and "mlp.shared_expert." in name
):
    # Map mlp.shared_expert.xx_proj to mlp.experts.{num_experts}.xx_proj
    name = name.replace(
        "mlp.shared_expert.",
        f"mlp.experts.{num_experts}.",
    )
```

## 性能声明

测试条件:MI355X,amd/Qwen3.5-397B-A17B-MXFP4,TP=2,EAGLE MTP(3 steps / topk 1 / 4 draft tokens),page-size 16,fp8_e4m3 KV,`--attention-backend aiter`,ISL=8192/OSL=1024,前后背靠背同 GPU 对(commit 7f8f030000)。总吞吐 conc=4/8/16 分别 +14.14%/+11.57%/+12.53%(4746.11→5417.31、6835.64→7626.54、9011.82→10141.08);TTFT 持平,TPOT 改善 9–12%。accept rate 由约 0.51 恢复到约 0.65;加载时每 TP rank 被跳过的权重由 3 降为 0。作者称恢复到此前 InferenceX 公布水平(602.47/857.12/1123.38 output tok/s vs 582.84/852.30/1130.14)。

## 验证与回归风险

每个并发点约 100 个 decode-batch 样本测 accept_len/accept_rate;预期无精度变化(投机草稿不能改变被验证的输出)。对 759 个 commit 二分 accept len 定位到 #33889,前后取值无中间值,证据链完整。回归风险限于 `_use_aiter` 且发生 shared-expert 融合的 AMD 配置;非融合配置下 `num_fused_shared_experts=0`,行为不变。

## 关联

前序 PR #33889(将 `disable_shared_experts_fusion` 改为 per-runner,是暴露本问题的变更)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
