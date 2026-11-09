# PR #11712: enable ut test for xpu devices

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DiweiSun |
| 合并 | 2026-02-03 |
| 分类 | Other |
| Label | documentation, high priority, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, intel, npu, xpu, run-ci, model-gateway |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含大量文件(约 20 个),另有 5 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/11712 |

## 一句话结论

把 SGLang 单元测试基础设施从硬编码 CUDA 泛化为多硬件(get_device()),使 UT 可在 Intel XPU 上运行,提升 XPU 后端的可验证性。

## 动机

SGLang 的单测大量硬编码 `device="cuda"` / `.cuda()` / `torch.cuda.*`,导致 Intel XPU 设备无法运行任何 UT,XPU 后端(经 `docker/Dockerfile.xpu` 构建的环境)的回归只能靠人工验证。作者声明该 PR scope 很大、按 review 要求可拆分为更小的 PR。

## 优化点

三方面改造。其一,test 基础设施:`test/runners.py` 中所有 `.cuda()` / `.to("cuda")` / `device="cuda"` 改为 `get_device()`;`test/test_utils.py` 新增跨硬件工具 `get_gpu_count()`(`torch.accelerator.device_count`)、`empty_gpu_cache()`(PyTorch 2.8 无 `torch.accelerator` 时回退 `torch.cuda`/`torch.xpu`)、`get_gpu_memory_gb()`(CUDA 用 `device_memory_used`,XPU 用 `memory_allocated`)。其二,逐测试泛化:`test_create_kvindices.py`、`test_wave_attention_kernels.py`、`test_causal_conv1d.py`、`test_fused_moe.py`(`create_random_cuda_tensor` 更名 `create_random_gpu_tensor`)、`test_awq_dequant.py`、`test_mamba_unittest.py`、`test_swa_unittest.py`、`test_gptqmodel_dynamic.py` 等把设备硬编码换成 `get_device()`,并移除 CUDA-only 的 skip 守卫。其三,运行时接线:`layers/moe/topk.py` 中 `topk_softmax` 内核的 import 条件从 `_is_cuda or _is_hip` 扩为 `_is_cuda or _is_hip or _is_xpu`,使 XPU 也能走 sgl_kernel 的 MoE topk 路径;`test_expert_location_updater.py` 增加 `torch.xpu.set_device(f"xpu:{rank}")` 分支。

## 关键改动

```text
python/sglang/test/test_utils.py: 新增 get_gpu_count / empty_gpu_cache / get_gpu_memory_gb 跨硬件工具
python/sglang/test/runners.py: .cuda() 全部改为 .to(get_device())
python/sglang/srt/layers/moe/topk.py: topk_softmax import 条件加入 _is_xpu
test/registered/*(attention/moe/quant/radix_cache/mamba 等): 设备硬编码泛化为 get_device()
```

## 性能声明

PR 未注明任何性能数字(定位为测试基础设施改造,非运行时优化)。topk_softmax 对 XPU 的启用属于功能接线,PR 未提供 XPU 上的 MoE 性能数据。

## 验证与回归风险

验证方式:在 XPU 环境构建(`docker/Dockerfile.xpu`)后 `pytest -v test_*.py`;部分用例需手动安装 pypi 包(flashinfer-python、sentencepiece、ray、accelerate、nest-asyncio)。风险面:改动面极宽(大量测试文件 + topk.py 运行时分支),对 CUDA 路径应是等价替换(`get_device()` 在 CUDA 上返回 "cuda");`get_gpu_memory_gb` 对非 CUDA/XPU 平台返回 0,可能掩盖某些平台的显存断言;`torch.accelerator` API 依赖较新 PyTorch(代码中显式处理 2.8/2.9 差异)。

## 关联

同区域:XPU/Intel 后端与 sgl-kernel 多硬件支持(labels 含 intel、xpu、npu、sgl-kernel)。区域内无其他已索引 PR 编号,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
