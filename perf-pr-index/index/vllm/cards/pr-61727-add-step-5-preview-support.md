# PR #61727: Add Step-5 Preview support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-10-19 |
| 分类 | Other (tier: major) |
| Label | documentation, new-model, speculative-decoding, ready, torch.compile, ci/build, tool-calling, cpu, kv-connector, nvidia, mrv2, rust, scheduler, kv-cache-manager, mooncake |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用,且注明 75 个文件因超长未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/61727 |

## 一句话结论

为 vLLM 增加 Step-5-Preview(stepfun-ai)模型支持,并在 MoE topk softmax CUDA kernel 中为 num_experts=352 的 sigmoid 路径新增专用 fast path。

## 动机

PR 正文为空模板(Overview/Claims/Validation 均未填写),未包含动机陈述。从 diff 可见目标是接入 `Step5ForConditionalGeneration`(Step-5-Preview,`stepfun-ai/Step-5-Preview` 等 checkpoint),模型登记进 supported_models.md(T + I+ 多模态);配套为 Rust 前端补 Step3-VL/Step5-VL 视觉 spec,并为其专家数补 topk kernel 路径。

## 优化点

新模型支持是主体;与性能索引相关的机制改动在 `csrc/moe/topk_softmax_kernels.cu`:新增 `case 352` 分支——仅 sigmoid scoring 时按每 lane 11 次标量加载(32 的倍数)launch 专用 topk 配置,否则落入通用 workspace 路径;同时把 switch 从 break 改为 return、将 256/512 等 2 的幂 case 移出 `#ifndef USE_ROCM` 保护(明确注释:非 2 幂 fast path 为 CUDA-only,ROCm 因 64-lane warp 不适用),default 分支(softmax/sigmoid + moeTopK 通用实现)移到 switch 之后统一执行。Rust 侧新增 `step3_vl` 视觉 spec 与 625 行 processor(728x728 全局视图 + 504x504 滑窗 patch,复刻 Python 版整数/截断语义),供 Step5-VL 前端复用。

## 关键改动

```text
vllm/models/step5/ 等(75 文件未收录,内容不可见): Step-5-Preview 模型实现
csrc/moe/topk_softmax_kernels.cu: case 352 sigmoid fast path + switch 重构
rust/src/chat/src/multimodal/models.rs + step3_vl/: Rust 前端 Step3-VL/Step5-VL 视觉 spec
docs/models/supported_models.md: 登记 Step5ForConditionalGeneration
```

## 性能声明

无性能声明;PR 正文为空模板,未包含任何 benchmark 数字或测试条件。case 352 fast path 相对通用 workspace 路径的收益未量化(通用路径需先写 workspace 再做 moeTopK,fast path 为单 kernel,但 PR 未给数据)。

## 验证与回归风险

Buildkite 新增 H200 Step5 step(`pytest models/step5`,依赖 `vllm/models/step5/` 与 `tests/models/step5/`);Rust 侧自带单测(spec 匹配、prompt replacement 展开、零 patch、field layouts)。风险面:topk kernel switch 重构触及所有 MoE 模型的 gating 路径——256/512 从 ROCm 专属保护中移出(ROCm 现在也走 POWER_OF_2 加载宽度),行为等价性依赖 64-lane warp 下的加载语义;75 个文件未收录,完整模型实现内容超出素材可见范围。

## 关联

同区域:同族模型 Step3-VL/Step-3.7-Flash 的既有支持;MoE topk kernel 的其他 expert 数 fast path(192/320/384/448/576,CUDA-only)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
