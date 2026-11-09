# PR #56985: [Bugfix][CPU] Add per-tensor FP8 W8A16 kernel to fix Ministral crash on CPU

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56985](https://github.com/vllm-project/vllm/pull/56985) |
| 作者 | bigPYJ1151 |
| 合并日期 | 2026-09-16 |
| labels | bug, ci/build, cpu, mistral, verified |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
CPU 上唯一的 per-tensor FP8 kernel 未接入 AMX 平台,fp8 量化的 Pixtral/Ministral-3-8B 模型加载即崩溃;新增基于 AMX block-scaled kernel 的 W8A16 per-tensor kernel 并注册到 CPU 后端解析列表。

## 问题现象
在具备 AMX 能力的 CPU 上加载 fp8 量化的 `PixtralForConditionalGeneration` 模型(例如 Ministral-3-8B-Instruct-2512)时崩溃(crash on load)。原因是没有可选的 per-tensor FP8 CPU kernel。此外正文还记录了排查过程中发现的两个 CPU CI flakiness(Buildkite #89028:多模态测试每次重新拉取视频资产;#89045:权重缓存不完整时 CPU spec-decode 直接失败而不回退在线重试)。

## 根因
PR 正文披露:`PerTensorTorchFP8ScaledMMLinearKernel` 是当时唯一可用的 per-tensor FP8 CPU kernel,但没有为 AMX-capable CPU 接线(wasn't wired up),导致 fp8 量化模型在 CPU 上加载时找不到匹配 kernel 而崩溃。

## 修复方案
1. `vllm/model_executor/kernels/linear/scaled_mm/cpu.py` 新增 `CPUFp8PerTensorScaledMMLinearKernel`(FP8 W8A16 per-tensor GEMM,复用既有 AMX block-scaled kernel `fp8_scaled_mm_cpu`,用单个覆盖整个权重的合成 block,scale 展开 `(1, num_k_blocks)`,激活保持 BF16/FP32、不做 FP8 激活量化);`can_implement` 要求 per-tensor scale、bfloat16/float32 输出、N 为 32 的倍数(AMX tinygemm 按 32 分块);`process_weights_after_loading` 把 [K,N] 转回 [N,K] 并 `convert_weight_packed` 预打包;
2. `kernels/linear/__init__.py` 与 `scaled_mm/__init__.py`:把新 kernel 注册进 `PlatformEnum.CPU` 的后端解析列表(排在 `PerTensorTorchFP8ScaledMMLinearKernel` 之前)与 `__all__`;
3. CI(`.buildkite/scripts/hardware_ci/run-cpu-test.sh`):`OFFLINE_RETRY_PATTERN` 增加 `Cannot find any model weights with`(匹配 default_loader.py 对不完整 HF cache snapshot 抛的 RuntimeError);docker 增加 `-v ~/.cache/vllm:/root/.cache/vllm` 挂载使多模态资产缓存跨容器保留;
4. `tests/v1/e2e/test_cpu_spec_decode.py` 模型名 `Qwen/Qwen3.5-0.8B-Base` → `Qwen/Qwen3.5-0.8B`。

```python
class CPUFp8PerTensorScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """FP8 W8A16 per-tensor-scaled GEMM via AMX BRGEMM on CPU. ..."""
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cpu():
            return False, "requires CPU platform."
        if not torch.cpu._is_amx_tile_supported():
            return False, "requires AMX tile support (Sapphire Rapids or newer)."
        ...
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:CPU 平台 per-tensor FP8 kernel 的选择顺序变化——AMX CPU 上会优先命中新 kernel(激活不再走 torch FP8 量化路径);非 AMX CPU、其他量化方式不受影响;N 非 32 倍数的层会由 `can_implement` 拒绝(有专门测试)。CI 脚本改动影响 CPU CI 的缓存挂载与离线重试匹配。验证:`test_cpu_fp8_scaled_mm.py` 80 passed(含 N 非 32 倍数拒绝路径)。

## 关联
正文未列出 GitHub issue/PR 链接(仅文字提及 Buildkite builds #89028、#89045)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
