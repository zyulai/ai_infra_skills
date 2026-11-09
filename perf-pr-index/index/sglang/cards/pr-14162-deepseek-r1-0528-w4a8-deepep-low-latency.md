# PR #14162: DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xieminghe1 |
| 合并 | 2026-03-30 |
| 分类 | MoE(major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/14162 |

## 一句话结论

DeepSeek-R1-0528-W4AFP8 的 DeepEP low-latency dispatch 从 BF16 改为 FP8 通信,sharegpt 负载吞吐约 +10%。

## 动机

profiling 显示:开启 DeepEP 后,DeepSeek-R1-0508-W4AFP8 模型的通信延迟是 DeepSeek-R1-0528 模型的两倍。根因是该 W4AFP8 模型的 DeepEP Dispatch 用 BF16 传输激活,带宽消耗翻倍,拖累推理性能;而通信数据最终会被量化成 FP8 喂给 cutlass_w4a8 MoE,先 BF16 传再量化纯属浪费。

## 优化点

两处配套改动。第一,通信侧:`token_dispatcher/deepep.py` 的 `_dispatch_core` 中,low-latency dispatch 判定从"仅当 `SGLANG_DEEPEP_BF16_DISPATCH` 未设置时才用 FP8"改为 else 分支默认 `use_fp8 = True`,即 W4AFP8 模型 DeepEP LL dispatch 直接以 FP8 发送(per-token 量化格式)。同时 `ep_moe/layer.py` 把"必须 BF16 dispatch"的断言从 `forward_cutlass_w4afp8_masked`(LL 路径)移到 `forward_cutlass_w4afp8`(normal 路径),即 normal dispatch 仍要求 BF16、LL dispatch 放开为 FP8。第二,计算侧:原来 LL 路径收到 BF16 激活后用 `per_tensor_quant_fp8` 整体重量化;现在收到的是 FP8 per-token 量化数据,新增 Triton kernel `fp8_per_token_to_per_tensor_quant_triton`,按 `masked_m` 屏蔽无效 token,把 per-token FP8 激活乘以自身 scale、再除以目标 per-tensor scale,直接反量化重标定成 per-tensor FP8(`float8_e4m3fn`),输出给 cutlass_w4a8 MoE 的 GEMM 接口。`cutlass_w4a8_moe_deepep_ll` 签名从单个 `a` 改为 `a_states + a_scales`,删除中间 BF16 量化步骤。

## 关键改动

```text
python/sglang/srt/layers/moe/token_dispatcher/deepep.py: LL dispatch 默认 FP8
python/sglang/srt/layers/moe/ep_moe/kernels.py: 新增
  fp8_per_token_to_per_tensor_quant_triton(per-token FP8 -> per-tensor FP8)
python/sglang/srt/layers/moe/cutlass_w4a8_moe.py: deepep_ll 接受 a_states+a_scales,
  用新 kernel 替代 per_tensor_quant_fp8
python/sglang/srt/layers/moe/ep_moe/layer.py: BF16 断言移到 normal dispatch
python/sglang/srt/layers/quantization/w4afp8.py: apply_deepep_ll 透传 hidden_scales
```

## 性能声明

测试条件:GPU H20,DeepSeek-R1-0528-W4AFP8,TP8/DP8 + dp-attention,DeepEP low_latency + cutlass MoE runner,EAGLE 投机(num_steps 1, topk 1, draft tokens 2),context 16384,`bench_serving` sharegpt 数据集 100 prompts、sharegpt-output-len 256、max-concurrency 64、request-rate 10。正文结论:改后吞吐约提升 10%("Throughput improved by approximately 10% after this pr"),具体数字以正文截图呈现(素材中无文本数值)。动机部分的 profiling 图显示 W4AFP8 开 DeepEP 后通信延迟约为 R1-0528 的两倍。

## 验证与回归风险

精度:gsm8k 与 ceval 两个数据集的 main vs 本 PR 对比结果均以截图给出,正文未提供具体数值。Checklist 全部未勾选,pack diff 未含新增单测。风险:LL dispatch 默认 FP8 改变了所有走 `SGLANG_DEEPEP_NORMAL_BF16_DISPATCH` 相关语义的调用方(该环境变量名在启动命令中仍被设置,但 LL 路径已不再读它),非 W4AFP8 模型若依赖 LL BF16 dispatch 会受影响;新量化 kernel 有硬约束(hidden 维度需被 128 与 1024 整除、x 连续、`output_scale` 为标量),不满足时 assert 失败;normal dispatch 路径仍强制 BF16,若误用 FP8 normal dispatch 会在 assert 处报错。

## 关联

后继:#22316(本 PR 的 Reland)。同区域:#25540(未量化 DeepEP 用 DeepGEMM BF16)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
