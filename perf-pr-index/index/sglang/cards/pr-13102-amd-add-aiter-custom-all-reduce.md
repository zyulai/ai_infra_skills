# PR #13102: [AMD] Add AITER Custom All-Reduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw |
| 合并 | 2025-11-13 |
| 分类 | Attention / Kernel | tier: major |
| Label | performance, amd, run-ci |
| Diff | diffstat 缺失(素材未提供统计数字);diff 含 7 个文件,新增 benchmark 脚本约 330 行 |
| 链接 | https://github.com/sgl-project/sglang/pull/13102 |

## 一句话结论

在 ROCm/AMD 上默认启用 AITER custom all-reduce,大消息 all-reduce 时延明显下降。

## 动机

PR 是 #11484 工作的延续,目标是在 AMD/ROCm 平台上用 AITER 提供的 custom all-reduce kernel 替代 SGLang 自带的 sgl-kernel CustomAllReduce 路径。TP 多卡推理中 all-reduce 是每个 layer 的固定开销,对 ROCm 用户此前只有 sgl-kernel 一条实现可选。PR 还提到 aiter 构建依赖 ROCm/aiter#1387 这个补丁。

## 优化点

核心是引入一个分发函数 `dispatch_custom_allreduce()`:在 HIP(ROCm)平台上且开关打开时,动态 import `aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce` 并作为 `ca_comm` 的实现类;import 失败(可选依赖缺失)则回退到原有 sglang `CustomAllreduce`,非 HIP 平台行为不变。开关通过环境变量控制,默认开启,可设 0 走回原 sgl-kernel 路径。另外给 `qr_rocm_arch_available()` 加了 `@cache`,避免重复探测 ROCm 架构。并行状态初始化处把硬编码的 `CustomAllreduce` 替换为按上述分发得到的类,使 aiter 实现无缝接入现有 custom all-reduce 调度。

## 关键改动

```text
python/sglang/srt/distributed/device_communicators/custom_all_reduce.py: 新增 dispatch_custom_allreduce(),按平台/环境变量选 aiter 或原生实现
python/sglang/srt/distributed/parallel_state.py: 初始化 ca_comm 时改用 dispatch_custom_allreduce() 返回的类
benchmark/kernels/all_reduce/benchmark_aiter.py: 新增脚本,对比 SGLang 与 Aiter custom AR 各消息尺寸时延
test/srt/test_custom_allreduce.py: 测试尺寸加入 67108864(64MB),补 global server args 设置
python/sglang/srt/distributed/device_communicators/quick_all_reduce.py: qr_rocm_arch_available() 加 @cache
```

关键片段:

```python
def dispatch_custom_allreduce():
    """Return the CustomAllreduce class to use (aiter on ROCm if enabled)."""
    if is_hip() and get_bool_env_var("SGLANG_USE_AITER_AR", default="true"):
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )
            logger.info("Using AiterCustomAllreduce for ROCm.")
            return AiterCustomAllreduce
        except ImportError as e:
            return CustomAllreduce
    return CustomAllreduce
```

注:PR 正文将开关写作 `SGLANG_AITER_AR`(默认 True),diff 代码中的变量名为 `SGLANG_USE_AITER_AR`,两处以素材原样记录。

## 性能声明

TP=8 下 `benchmark/kernels/all_reduce/benchmark_aiter.py` 对比(单位 ms,SGLang vs Aiter):小消息上 Aiter 略慢或持平(32K:0.038 vs 0.045;1M:0.058 vs 0.053),大消息上 Aiter 明显更快——16M:0.167→0.130,32M:0.270→0.201,64M:0.508→0.365。PR 未注明该 benchmark 的具体 GPU 型号与精度(脚本内使用 float16)。

准确性测试:DeepSeek-R1-MXFP4-Preview,TP=8 + EAGLE speculative(num_steps=3, topk=1, draft_tokens=4),gsm8k 1400 题 parallel 1400:Accuracy 0.942,Invalid 0.000。

## 验证与回归风险

依赖可选依赖 aiter(需 ROCm/aiter#1387 补丁),import 失败自动回退原生实现,因此对非 AMD 环境路径无影响。test_custom_allreduce.py 增加了 64MB 尺寸并需设置 global server args;run_suite.py 中临时禁用了 lora/test_lora_backend.py 等(关联 issue #13107,属临时处置);test_programs.py 放宽了 HIP 平台的一项 latency 断言阈值(1→2)。风险点:ROCm 上默认行为改变(默认走 aiter),若 aiter 版本不带所需补丁可能回退或行为异常。

## 关联

前序:#11484(本 PR 为其延续)。同区域:custom all-reduce/通信相关 PR 如 #11381(symm-mem allreduce)、#13797(all-reduce benchmark 差异)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
