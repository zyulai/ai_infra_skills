# PR #43373: [diffusion] perf: pipeline MiniMax-H3's Ulysses exchange against cuDNN SDPA on Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-10 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, run-ci, diffusion |
| Diff | pack 未提供 diffstat（diff 内容可用：2 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/43373 |

## 一句话结论

MiniMax-H3 的 Ulysses 分组流水线交换放行 cuDNN SDPA 后端，4x B300 e2e 时延缩短 3.4-4.3%（Hopper 上为 2-3%）。

## 动机

Blackwell 上 MiniMax-H3 的 dense attention 走 cuDNN SDPA（`DynamicCudnnSDPAImpl`）。但 #43164 引入的 Ulysses 流水线交换只接受 FlashAttention 后端，B200/B300 只能继续用顺序交换，完全享受不到"上一组 head 的 attention 与下一组 head 的 q/k/v 拷贝重叠"的收益。

## 优化点

机制上是放开门控：`_minimax_h3_pipelined_dense_attention` 的后端白名单从仅 `FA` 扩为 `FA + DYNAMIC_CUDNN_SDPA`（`_PIPELINED_ATTENTION_BACKENDS`）。attention 仍经后端 `forward_varlen` 逐 head group 执行；`_request_skip_softmax_threshold` 检查只存在于 FlashAttention impl，故改为可选（`getattr` 判空，cuDNN impl 无此方法）。前提是"分 head group 调用 cuDNN SDPA 与整头调用的输出逐字节相同"，作者在 H3 形状上验证。同时更新 cookbook 说明：流水线在 Hopper 上对 FlashAttention、Blackwell 上对 cuDNN SDPA 生效。本 PR 自身改动即 cuDNN 门控 + 一行 cookbook（stack 在 #43361/#43164 之上）。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py: pipelined dense attention 白名单加入 DYNAMIC_CUDNN_SDPA，skip-softmax 检查改可选
docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx: 流水线适用后端说明更新
```

## 性能声明

4x B300，50 steps，default quality，每 arm 单次运行，client e2e（基线为 #43164+#43361 栈，`SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS` 选 0=顺序或组数）：T2VA 5s Ulysses2 65.83 -> 63.24 s（-3.9%，7 groups）；Ulysses4 34.20 -> 32.75 s（-4.3%）；Ref2VA 5s Ulysses2 102.68 -> 99.24 s（-3.4%，4 groups 时 -3.9%）。Blackwell 增益大于 Hopper（2-3%），因为交换在更快的 step 中占比更大。分组的代价：B300 上 7 组调用比整头单次慢 3.5%（Ref2VA 形状）/ 5.4%（Ulysses4 形状）；7 vs 4 组的 ABBA 对比只差 0.1-0.2%（噪声内），故自动组数保持 7。

## 验证与回归风险

逐 head group 的 cuDNN SDPA 输出与整头调用 `torch.equal` 验证（4x B300，三种 H3 形状：49652/37742/37744 tokens，28 heads 组 4 或 14 heads 组 2；torch 2.14.1+cuDNN 9.24 与 torch 2.13+cuDNN 9.20 都成立）；e2e MP4 md5 在顺序交换与 7/4/2 组下全部相同。风险：cuDNN 分组调用的效率损失是 FA3 不付的（组数权衡已在测量中覆盖但仅单次运行 + 一组 ABBA）；门控依赖 `attention._attention_backend_enum` 与 `forward_varlen` 的后端契约，其他后端仍走顺序交换；未新增单测（流水线 parity 测试本身是 backend 无关的）。

## 关联

Stack 关系：#43164（Ulysses 流水线交换本体，仅 FA）、#43361（前序 stack）、#43327/#43337（性能基线栈内）；本 PR 是在 Blackwell 上补齐 cuDNN 后端支持。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
