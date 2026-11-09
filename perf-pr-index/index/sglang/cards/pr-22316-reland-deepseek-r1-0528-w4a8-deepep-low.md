# PR #22316: [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xieminghe1 |
| 合并 | 2026-04-10 |
| 分类 | MoE(major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/22316 |

## 一句话结论

Reland #14162:W4AFP8 模型 DeepEP LL dispatch 由 BF16 改 FP8 通信,sharegpt 负载吞吐约 +10%。

## 动机

与 #14162 相同:profiling 显示开启 DeepEP 后 DeepSeek-R1-0508-W4AFP8 的通信延迟是 DeepSeek-R1-0528 的两倍,根因是其 DeepEP Dispatch 用 BF16 传输激活,带宽消耗翻倍。本 PR 为 #14162 的重新合入(reland),正文未说明原合入被回退的原因;PR 头部的 Motivation/Modifications 段为空,完整内容在正文中重复贴出。

## 优化点

与 #14162 完全一致,两个配套改动。通信侧:`deepep.py` 的 `_dispatch_core` 中,low-latency dispatch 从"仅当 `SGLANG_DEEPEP_BF16_DISPATCH` 未设置时才用 FP8"改为 else 分支默认 `use_fp8 = True`,W4AFP8 模型的 DeepEP LL dispatch 直接以 FP8(per-token 量化格式)发送;`ep_moe/layer.py` 的 `forward_cutlass_w4afp8_masked` 删除"必须 BF16 dispatch"断言。计算侧:新增 Triton kernel `fp8_per_token_to_per_tensor_quant_triton`,按 `masked_m` 屏蔽无效 token,把 per-token FP8 激活乘自身 scale、除以目标 per-tensor scale,直接转换成 per-tensor FP8 输出;`cutlass_w4a8_moe_deepep_ll` 签名改为 `a_states + a_scales`,删除原 BF16 输入的 `per_tensor_quant_fp8` 中间重量化步骤,`w4afp8.py` 的 `apply_deepep_ll` 相应透传 `hidden_scales`。

## 关键改动

```text
python/sglang/srt/layers/moe/token_dispatcher/deepep.py: LL dispatch 默认 FP8
python/sglang/srt/layers/moe/ep_moe/kernels.py: 新增
  fp8_per_token_to_per_tensor_quant_triton(per-token FP8 -> per-tensor FP8)
python/sglang/srt/layers/moe/cutlass_w4a8_moe.py: deepep_ll 接受 a_states+a_scales
python/sglang/srt/layers/moe/ep_moe/layer.py: 移除 masked 路径的 BF16 断言
python/sglang/srt/layers/quantization/w4afp8.py: 透传 hidden_scales
```

## 性能声明

与 #14162 相同的测试条件:GPU H20,DeepSeek-R1-0528-W4AFP8,TP8/DP8 + dp-attention,DeepEP low_latency + cutlass MoE runner,EAGLE 投机(steps 1, topk 1, draft tokens 2),context 16384,`bench_serving` sharegpt 100 prompts、output-len 256、max-concurrency 64、request-rate 10。正文结论:改后吞吐约 +10%,具体数值以截图呈现(素材中无文本数值)。动机 profiling 图显示 W4AFP8 开 DeepEP 后通信延迟约为 R1-0528 的两倍。

## 验证与回归风险

精度:gsm8k 与 ceval 的 main vs 本 PR 对比均以截图给出,正文无具体数值;Checklist 全部未勾选,pack diff 未含新增单测。与原 PR 相同的风险:LL dispatch 默认 FP8 后,非 W4AFP8 模型若依赖 LL BF16 dispatch 会受影响;新量化 kernel 对 hidden 维度有硬约束(需被 128 与 1024 整除、x 连续、output_scale 为标量),不满足时 assert 失败。作为 reland,额外风险是需确认当初回退的原因在当前代码基上已消除,但 PR 正文未提供相关信息。

## 关联

前序:#14162(首次合入,本 PR 为其 reland)。同区域:#25540(未量化 DeepEP 用 DeepGEMM BF16)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
