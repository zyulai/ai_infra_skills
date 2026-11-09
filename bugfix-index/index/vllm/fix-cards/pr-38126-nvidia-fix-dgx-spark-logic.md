# PR #38126: [NVIDIA] Fix DGX Spark logic

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38126](https://github.com/vllm-project/vllm/pull/38126) |
| 作者 | johnnynunez |
| 合并日期 | 2026-03-27 |
| labels | ready, ci/build, nvidia |
| 热度 | 👍 8 · 💬 22 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论

SM121(DGX Spark)构建时 arch guard 只认 12.0 系列导致 NVFP4/scaled_mm/MLA 等 SM12x 内核被静默跳过;补充 12.1 变体并修复后缀匹配逻辑。

## 问题现象

用 `TORCH_CUDA_ARCH_LIST=12.1a` 构建 vLLM 时,构建日志显示所有 SM12x 内核被跳过:

```
Not building scaled_mm_c3x_120 as no compatible archs found
Not building NVFP4 as no compatible archs were found.
Not building CUTLASS MLA as no compatible archs were found.
```

运行时在 DGX Spark 上启动 NVFP4 模型会报 `NotImplementedError: No compiled nvfp4 quantization kernel`。影响面是所有 SM121/GB10(DGX Spark)上从源码构建的用户。

## 根因

正文披露:PR #37725 正确保留了 gencode 提取时的 CUDA arch 后缀(`a`/`f`),但 `CMakeLists.txt` 下游的 arch guard 检查只识别 `12.0a`/`12.0f`,不匹配 `12.1`/`12.1a`/`12.1f`(SM121/DGX Spark),导致以 `12.1a` 构建时所有 SM12x 家族内核被静默跳过。另外 `cmake/utils.cmake` 的 `cuda_archs_loose_intersection` 存在两个缺口:TGT 带后缀而 SRC 无后缀时(TGT=`12.1a` + SRC=`12.1`)输出丢了后缀;SRC 与 TGT 后缀不同时(SRC=`12.1f` + TGT=`12.1a`)输出为空。guard 使用 `"12.0f"`/`"12.0a"` 这类 SRC 列表做精确基版本匹配,目标为 `12.1` 时即失败。

## 修复方案

1. `cmake/utils.cmake`:`cuda_archs_loose_intersection` 增加跨后缀匹配(SRC `x.yf` + TGT `x.ya`,同基版本时用 TGT 后缀输出)与 `f` 家族后缀回退(SRC `12.0f` 可匹配同主版本家族的 TGT `12.1a`);并做对称处理——TGT 带 `x.ya/f` 而 SRC 为 `x.y` 时保留 TGT 后缀输出。
2. `CMakeLists.txt`:为所有 SM12x 家族 guard 增加 `12.1` 变体,如 Marlin FP8/MOE Marlin FP8 改为 `"8.9;12.0;12.1"`,scaled_mm 与 NVFP4 在 CUDA>=13.0 时用 `"12.0f;12.1f"`、否则用 `"12.0a;12.1a"`。摘录:

```diff
-  cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
+  cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0;12.1" "${CUDA_ARCHS}")
```

3. `cmake/external_projects/qutlass.cmake`:arch 列表加入 `12.1a`,TARGET_CC 正则改为 `12\.[01][af]?` 以匹配 SM121。

## 影响与回归风险

- 影响窗口:PR #37725 合并之后至本修复之前的构建;正文未给出具体版本号。
- 回归面:CMake 构建配置改动,影响所有 CUDA 平台的内核编译选择——Marlin FP8/MOE、scaled_mm、NVFP4、CUTLASS MLA、QuTLASS、moe_data 等 guard;SM120(RTX 5090)等既有 arch 的匹配结果依赖新加的跨后缀/家族回退逻辑,若匹配规则变化可能改变其他 arch 的编译集合。作者在 test plan 中要求 `12.0a` 构建无回归来验证。

## 关联

- #37725 — Preserve CUDA arch suffix (a/f) for SM12x(已合并,引入本回归)
- #34822 — `is_blackwell_class()` for SM121/GB10 DGX Spark support
- #37700 — FLA Hopper/TMA misclassification on SM12x desktop Blackwell

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
