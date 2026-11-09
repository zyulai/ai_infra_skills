# PR #20358: Update PyTorch to 2.8.0

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @huydhn |
| 合并 | 2025-08-29 |
| 分类 | Other |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/20358 |

## 一句话结论

把 vLLM 的 PyTorch 依赖从 2.7.1 升级到 2.8.0(CUDA 与 ROCm 同步),并带动 triton/xformers/NCCL 等全套依赖更新。

## 动机

PyTorch 2.8.0 正式发布,vLLM 需跟进以获得新版 torch 的能力与修复。升级涉及全仓库依赖与 CI,作者逐一确认各 test 套件的失败是否为 main 上已存在的失败,并用 PyTorch 官方 benchmark 验证无性能回归。

## 优化点

机制上是依赖基线升级:torch 2.7.1 → 2.8.0,`TORCH_SUPPORTED_VERSION_CUDA`/`TORCH_SUPPORTED_VERSION_ROCM` 统一为 2.8.0;torchvision 0.23.0、torchaudio 2.8.0、xformers 0.0.32.post1、triton 3.4.0、NCCL 2.27.3 等随 torch 一起更新;ROCm 侧索引从 rocm6.2.4 切到 rocm6.3。CPU 平台版本重排:Darwin/ppc64le/aarch64 升到 2.8.0,但 x86_64 仍 pinned `torch==2.6.0+cpu`(注释:>2.6.0+cpu 在 x86 平台有性能回归,见 pytorch/pytorch#151218)。一处内核配置修正:flex_attention 后端在 shared memory 不足 144KB 时将 `BLOCK_M`/`BLOCK_N` 减半,而不是硬编码为 32。另因 NCCL >= 2.26.3 占用更多 GPU 显存,为 chatglm3 TP4 fully-sharded LoRA 测试调低 `gpu_memory_utilization` 至 0.85。

## 关键改动

```text
requirements/cuda.txt, test.txt: torch 2.8.0、triton 3.4.0、xformers 0.0.32.post1、NCCL 2.27.3
requirements/rocm-build.txt: torch 2.8.0 + rocm6.3 索引
CMakeLists.txt: TORCH_SUPPORTED_VERSION_CUDA/ROCM = 2.8.0
vllm/v1/attention/backends/flex_attention.py: 低 shared memory 设备 BLOCK_M/BLOCK_N 减半
tests/lora/test_chatglm3_tp.py: NCCL 显存增长导致的 OOM workaround
```

```python
if max_shared_memory < 144 * 1024:
    kernel_options["BLOCK_M"] = kernel_options["BLOCK_M"] // 2
    kernel_options["BLOCK_N"] = kernel_options["BLOCK_N"] // 2
```

## 性能声明

PR 正文未给出具体性能数字。作者引用 PyTorch benchmark hud 对 vllm-project/vllm 的对比(CUDA, NVIDIA H100 80GB HBM3, 2025-08-01 至 08-08 区间)称结果 "look normal";CI 各失败项均判定为 main 已存在的失败。数字未以文字给出,PR 未注明测试条件细节。

## 验证与回归风险

验证依赖 Buildkite 全量 CI(basic-models、blackwell、distributed 2/4-gpus、language-models-hybrid、multi-modal、v1-test 等),作者逐项核对失败均为 main 既有失败。回归面覆盖所有模型与硬件路径:NCCL 2.27.3 显存增长已实际引发 TP4 fully-sharded LoRA 测试 OOM(以调低 gpu_memory_utilization 规避);x86 CPU 因 torch 2.6 以上版本性能回归而保持 pinned;`mamba_ssm` 包破损影响 hybrid 模型测试信号(PR #22409 跟进)。

## 关联

无(依赖基线升级,与本批其他单点优化无直接前序/后续关系)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
