# PR #30700: [NVIDIA] Add flashinfer MNNVL backend for allreduce only

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenscarl |
| 合并 | 2026-08-11 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/30700 |

## 一句话结论

把纯(非融合)allreduce 也路由到 FlashInfer kAllReduce,复用已有 MNNVL workspace,DeepSeek V4 Flash TP4 小 batch decode 吞吐 +6.9%。

## 动机

SGLang 已为 DeepSeek-V3/V4 支持 FlashInfer 融合 allreduce(`kARResidualRMSNorm`:allreduce + residual + RMSNorm 单 kernel),但并非所有 allreduce 站点都能走融合路径——`RowParallelLinear`、attention 输出 fallback、MoE TP fallback 执行的是纯 allreduce,此前即使 FlashInfer mnnvl/trtllm workspace 已初始化可用,也总是回落到 NCCL/custom-allreduce。

## 优化点

新增 `flashinfer_allreduce()` 包装 `AllReduceFusionPattern.kAllReduce`,复用既有 workspace,无额外显存。分发机制:`_tag_groups_for_flashinfer_allreduce_only()` 在 `initialize_model_parallel()` 后给 `_ATTN_TP`/`_MOE_TP`/`_MOE_EP` 组协调器打 `_fi_workspace_hint` 标记(`_TP` 有意不打:它要么等于 `_ATTN_TP`,要么是其超集,attention workspace 对不上 peers);`GroupCoordinator.all_reduce()` 在 `ca_comm`/`pymscclpp`/NCCL 链之前检查 hint 并分发,不适配时透明回退。适用性判断拆在 custom op 外的 `can_use_flashinfer_allreduce()`(2D 连续、workspace 初始化/容量、group key 精确匹配——hybrid EP+TP 下 MoE-TP 与 MoE-EP world size 相同但配对不同 rank,错配会静默算错,故要求精确匹配);op 注册为 custom op 以在 Dynamo/piecewise CUDA graph 下不透明。同时修复 communicator 融合门:hybrid EP+TP 时 post-experts allreduce 跨两个不相交组,不能被融合吸收(Qwen3-30B-A3B tp=4 ep=2 观测到乱码输出)。DeepSeek-V3/V4 架构自动启用,无需手动传参。

## 关键改动

```text
python/sglang/srt/layers/flashinfer_comm_fusion.py: can_use_flashinfer_allreduce / flashinfer_allreduce(kAllReduce, launch_with_pdl)
python/sglang/srt/distributed/parallel_state.py: _fi_workspace_hint 标记与 all_reduce 分发钩子、custom op
python/sglang/srt/arg_groups/overrides.py: DeepSeek-V3/V4 自动启用 pure allreduce
python/sglang/srt/layers/communicator.py: hybrid EP+TP 禁用 mlp allreduce 融合的门控修复
test/registered/unit/layers/test_flashinfer_comm_fusion.py 等: guard/正确性/标记单测
```

## 性能声明

DeepSeek V4 Flash,Blackwell,`sglang.benchmark.one_batch`,TP=4,bs 1-64,input 512 / output 128,`--moe-runner-backend flashinfer_mxfp4`、`--disable-flashinfer-autotune`:

- Decode 吞吐:bs 1/2/4/8/16/32 分别 +6.9%/+6.8%/+6.4%/+3.0%/+2.6%/+3.0%;bs 64 为 -1.4%。bs 1:147.51 → 157.69 tok/s,decode 延迟 6.78 → 6.34 ms;overall 吞吐 bs 1 539.58 → 624.73 tok/s、bs 8 2863.78 → 4586.34 tok/s。
- 精度:GSM8K(8-shot,1316 题)0.958 → 0.958(Invalid 0.000);该评测的输出吞吐 1312.239 → 1158.934 token/s、延迟 95.622 → 107.229 s(PR 如实给出,未解释回退原因)。

## 验证与回归风险

新增单测:5 个 `_FakeFlashInferComm` 用例覆盖正确性(output == input x world_size)、1D/3D/非连续 guard、unavailable flag、workspace 未初始化;另有 group 标记测试(hybrid EP+TP 只标 EP 组、共享协调器确定性选 attn_tp)与融合门回归测试(CI base-c 4-gpu-h100/b200、base-a-test-cpu)。风险面:workspace 与 group 不匹配会静默产出错误数据(已用精确匹配防御);kernel 错误故意不捕获以免单 rank 回落 NCCL 造成 mismatch-hang;bs 64 有小幅回退;PR Test (Base)/(Extra) 均通过。

## 关联

前序:FlashInfer 融合 allreduce(`kARResidualRMSNorm`)路径与 DeepSeek-V3/V4 的 `--flashinfer-allreduce-fusion-backend` 支持(pack 未给出其 PR 编号);本 PR 与其在 `overrides.py` 中同一 auto-enable 体系。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
