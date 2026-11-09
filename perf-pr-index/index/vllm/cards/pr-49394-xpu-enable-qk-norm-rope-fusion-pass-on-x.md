# PR #49394: [XPU] Enable QK Norm + RoPE fusion pass on XPU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chaojun-zhang |
| 合并 | 2026-07-27 |
| 分类 | Attention / Kernel (tier: major) |
| Label | intel-gpu, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 3 个文件(Buildkite 配置、pass_manager、xpu 平台) |
| 链接 | https://github.com/vllm-project/vllm/pull/49394 |

## 一句话结论

在 XPU 上放开 QK Norm + RoPE fusion pass,Qwen3-8B 吞吐较 fusion_off 提升 11.7%。

## 动机

PR 正文未注明动机/瓶颈描述(正文仅给出 Performance & Accuracy Test)。从 diff 可看出:此前 `QKNormRoPEFusionPass` 与 `SplitCoalescingPass` 的 import 位于仅 CUDA 的平台分支,XPU 平台的 `check_and_update_config` 又把 `enable_qk_norm_rope_fusion` 列入 `fusion_passes_to_disable` 强制禁用,因此该 fusion 在 XPU 上无法启用。

## 优化点

把 `QKNormRoPEFusionPass` 与 `SplitCoalescingPass` 的 import 从 `current_platform.is_cuda_alike()` 分支上移到 `is_cuda_alike() or is_xpu()` 分支,使 XPU 也能装配该融合 pass;并从 `vllm/platforms/xpu.py` 的 `fusion_passes_to_disable` 字典中删除 `enable_qk_norm_rope_fusion` 条目,不再强制关闭。同时在 Buildkite 的 `.buildkite/intel_jobs/misc_intel.yaml` 新增 `Fusion Unit Tests` step,在 intel_gpu 设备上跑 `tests/compile/passes/test_qk_norm_rope_fusion.py`,并声明 `vllm/compilation/` 与该测试文件为 source_file_dependencies。改动均为平台开关/CI 装配,不涉及融合算子本身的数值实现。

## 关键改动

```text
vllm/compilation/passes/pass_manager.py: 把 QKNormRoPEFusionPass、SplitCoalescingPass 的 import 扩展到 is_xpu() 平台分支
vllm/platforms/xpu.py: 从 fusion_passes_to_disable 移除 enable_qk_norm_rope_fusion,不再在 XPU 上强制禁用
.buildkite/intel_jobs/misc_intel.yaml: 新增 Fusion Unit Tests 步骤,在 intel_gpu 上跑 test_qk_norm_rope_fusion.py
```

## 性能声明

PR 正文(Performance & Accuracy Test):服务 `vllm serve Qwen/Qwen3-8B ... -tp=1 --compilation-config '{"pass_config":{"enable_qk_norm_rope_fusion":true}}'`,压测 `vllm bench serve --model Qwen/Qwen3-8B --dataset-name random --random-input-len=4096 --random-output-len=1024 --num-prompt 32 --max-concurrency 8 --request-rate inf --backend vllm --temperature=0`。Benchmark(input=4096, output=1024, num-prompts=32, max-concurrency=8):fusion_on 321.33s / 101.98 output tok/s / 509.89 total tok/s / mean TTFT 17651.13 ms / mean TPOT 53.39 ms;fusion_off 358.83s / 91.32 / 456.60 / 23016.48 ms / 54.09 ms;eager baseline 321.58s / 101.90 / 509.48 / 16938.86 ms / 55.13 ms。正文结论:fusion_on 相对 fusion_off 给 **+11.7% throughput** 与 **-10.5% duration**,与 eager 持平。PR 未注明硬件型号(平台为 XPU/Intel GPU)。精度:GSM8K 5-shot limit=250,fusion_on 0.908/0.916(strict/flexible)、fusion_off 0.912/0.916、eager 0.904/0.912,均在 1 stderr 内,无精度回归。

## 验证与回归风险

PR 引入 Buildkite `Fusion Unit Tests` step(`pytest -v -s compile/passes/test_qk_norm_rope_fusion.py` on intel_gpu),并给出上述 Qwen3-8B 吞吐与 GSM8K 精度对比。影响路径:XPU 平台开启 compilation 时的 QK Norm + RoPE 融合装配;正文声明精度无回归,未列出其他模型/硬件的影响范围,也未注明具体 Intel GPU 型号。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
