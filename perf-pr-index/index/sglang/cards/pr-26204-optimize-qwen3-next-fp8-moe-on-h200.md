# PR #26204: Optimize Qwen3 Next FP8 MoE on H200

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-11 |
| 分类 | MoE |
| Label | run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/26204 |

## 一句话结论

为 Qwen3-Next FP8 MoE 开启 CUDA 共享专家融合并补齐 H200 Triton 调优配置,Chat 吞吐 +6%。

## 动机

共享专家融合(shared expert fusion)此前只在 AMD Aiter 路径(_use_aiter)启用,CUDA 上的 Qwen3-Next FP8 MoE 每层都要单独跑一次 shared expert 前向,与 routed expert 分离;且融合后专家数从 512 变为 513(含 1 个 fused slot),H200 上没有对应的 E=513 Triton MoE 调优配置,融合路径开不起来或跑不快。

## 优化点

把 shared expert fusion 的开关从 `_use_aiter` 泛化为模型/后端支持位:Qwen2MoeSparseMoeBlock 新增 `enable_cuda_shared_expert_fusion` 参数,`_is_cuda` 且模型 opt-in 时同样可融合;`can_fuse_shared_expert` 的量化排除层判断补上 `ignored_layers` 回退。Qwen3-Next(linear_attn 与 full attention 两种 MoE 层)传入 `support_shared_expert_fusion=True` + `enable_cuda_shared_expert_fusion=True`,保持对 Qwen2-MoE 默认行为不变。checkpoint 侧在 load_weights 中把 `mlp.shared_expert.*` 权重 remap 进 fused MoE expert 槽位(`mlp.experts.{num_experts}.`),只支持恰好 1 个共享专家(超出报错);MTP 路径镜像同样的 fusion 设置以保证权重加载一致。同时新增 H200 FP8 Triton MoE 配置:E=513,N=512、fp8_w8a8、block_shape [128,128] 的 normal 与 down 两份 JSON,down 配置各 batch size 均启用 TMA。

## 关键改动

```text
python/sglang/srt/models/qwen2_moe.py: get_num_shared_experts + enable_cuda_shared_expert_fusion 开关泛化
python/sglang/srt/models/qwen3_next.py: opt-in 融合,load_weights 将 shared_expert remap 进 experts 第 513 槽
python/sglang/srt/models/qwen3_next_mtp.py: MTP 层镜像 fusion 设置
.../triton_3_6_0/E=513,N=512,...H200..._down.json: H200 FP8 fused MoE 调优配置(down 含 USE_TMA)
```

## 性能声明

ion8-h200、单卡 NVIDIA H200、TP=1、Qwen/Qwen3-Coder-Next-FP8,sglang-sota-humanize-loop 随机 workload。Chat 1000/1000:SGLang baseline 3252.81 output tok/s、p50 TTFT 983.79 ms、p50 TPOT 23.51 ms;vLLM latest best 3246.42 / 1520.49 / 23.07;patched 3453.59 / 924.08 / 22.18(领先 vLLM)。Summarization 8000/1000:baseline 2084.86 / 6880.61 / 31.48(TTFT SLA fail);vLLM 2155.38 / 7288.98 / 29.38;patched 2140.12 / 6633.81 / 30.74,复测 2171.86 / 6636.10 / 30.19(与 vLLM 持平或领先)。TensorRT-LLM latest PyTorch backend 因该环境不支持 qwen3_next 未参与对比。

## 验证与回归风险

GSM8K 8-shot:baseline 0.954996,patched 初跑 0.945843(-0.009),current-head(ca4c18e)复测 0.952055(-0.003);MMLU full 0.874163 对 0.872098(-0.002);评测耗时同步下降(GSM8K 130.35s 到 105.58s,output tok/s 1343.17 到 1636.11)。临时验证测试(7 passed)在提交前移除,新配置 JSON 验证可解析。回归面:触及 qwen2_moe.py 的通用 MoE 代码(Qwen2/Qwen3 系列 MoE 模型共用),融合在 CUDA 上仅对显式 opt-in 的 Qwen3-Next 生效;权重 remap 只支持单共享专家;`shared_expert is None` 的分支被显式化,双流路径的 shared_output 需判空。

## 关联

同区域:Qwen3-Next/Qwen3.5 MoE 优化(如共享专家融合的 Aiter 先行实现、H200 Triton MoE config 系列);素材未给出具体编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
