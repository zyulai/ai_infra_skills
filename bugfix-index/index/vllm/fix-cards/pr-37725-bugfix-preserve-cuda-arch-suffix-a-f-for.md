# PR #37725: [Bugfix] Preserve CUDA arch suffix (a/f) for SM12x — fixes NVFP4 NaN on desktop Blackwell

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37725](https://github.com/vllm-project/vllm/pull/37725) |
| 作者 | RobTand |
| 合并日期 | 2026-03-25 |
| labels | bug, ci/build, nvidia, ready |
| 热度 | 👍 1 · 💬 19 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
cmake 构建丢掉 CUDA gencode 的 a/f 后缀,SM12x 编成 sm_120 导致 NVFP4 推理 NaN;修复为保留后缀并补充 12.1 支持。

## 问题现象
SM12x 设备(DGX Spark、RTX 5090)上 NVFP4 推理产生 NaN。vLLM 的 cmake 构建把架构专属后缀(`a`/`f`)从 CUDA gencode 标志中剥掉,使 SM12x 被编译成普通的 `sm_120` 而非 `sm_120a`/`sm_121a`。修复前 `cuobjdump -lelf _C.abi3.so` 显示的是无后缀的 `sm_120` cubin。

## 根因
正文披露:没有后缀时 `__CUDA_ARCH_FAMILY_SPECIFIC__` 不会被定义,NVIDIA 自家的 `cuda_fp4.hpp`(经 `__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__`)会禁用原生 `cvt.rn.satfinite.e2m1x2.f32` PTX 指令,回退到软件 E2M1 转换——后者在 NVFP4 推理中产生 NaN。构建层面的 bug 有两处(均在 `cmake/utils.cmake`):`string_to_ver` 的正则 `([0-9]+)([0-9])` 丢掉 `a`/`f` 后缀(`121a` 变成 `12.1`);`extract_unique_cuda_archs_ascending` 的正则 `[0-9]+a?` 只匹配 `a` 不匹配 `f`。另外 `CMakeLists.txt` 的 `CUDA_SUPPORTED_ARCHS` 缺少 `12.1`(SM121/DGX Spark)。

## 修复方案
- `cmake/utils.cmake` 的 `string_to_ver`:`([0-9])` 改为 `([0-9][af]?)`,保留后缀(`121a` → `12.1a`)。
- `extract_unique_cuda_archs_ascending`:`[0-9]+a?` 改为 `[0-9]+[af]?`,同时匹配两种后缀。
- `CMakeLists.txt`:CUDA 12.8 与 13.0 两档的 `CUDA_SUPPORTED_ARCHS` 均追加 `12.1`。
- 验证:修复后产出 `sm_121a` cubin 且 `__CUDA_ARCH_FAMILY_SPECIFIC__=1210`;在 DGX Spark 上确认原生 E2M1 PTX 正常(`PTX path enabled: 1`,float2(1.5, 0.5) → 0x13)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;影响使用带后缀架构(SM12x 及同类)的 cmake 构建产物,预编译 wheel 是否受影响正文未提。
- 回归面:仅构建系统(gencode 标志的解析与透传、支持架构列表);不改变任何运行时代码,但会改变扩展库的编译目标,其他依赖 `string_to_ver`/架构提取的构建逻辑需兼容带后缀的版本号。

## 关联
#35947(软件 E2M1 回退,本构建 bug 的 workaround), #34822(SM121 平台检测), #37700(同作者的 FLA Hopper/TMA 修复,独立问题),及 @depaulmillz 的指出评论

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
