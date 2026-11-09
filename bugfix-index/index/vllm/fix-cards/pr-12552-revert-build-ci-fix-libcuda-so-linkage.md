# PR #12552: Revert "[Build/CI] Fix libcuda.so linkage"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#12552](https://github.com/vllm-project/vllm/pull/12552) |
| 作者 | tlrmchlsmth |
| 合并日期 | 2025-01-29 |
| labels | ready, ci/build |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#12424 给 CUDA 扩展链接 libcuda,导致 LD_LIBRARY_PATH 无 libcuda 的用户出错;回滚该链接改动。

## 问题现象
正文披露:#12424 对一部分 `LD_LIBRARY_PATH` 上没有 `libcuda.so` 的用户造成问题(细节见 issue #12505)。作者还提到 `_C.abi3.so` 可能存在链接问题,但表示没见过用户实际踩到的具体例子。

## 根因
正文披露:#12424(Fix libcuda.so linkage)是问题来源——它让 vLLM 链接 `libcuda.so`,而部分用户环境中该库不在 `LD_LIBRARY_PATH` 上,导致出错(issue #12505)。作者认为对 `_C.abi3.so` 确实有潜在链接问题,但因缺乏具体用户案例,决定先回滚,后续再研究 PyTorch 的做法(通过 driver API 动态加载,正文引用了 pytorch c10 `driver_api.cpp` 的链接作为参考方向)。

## 修复方案
回滚 `CMakeLists.txt` 中 #12424 加入的链接配置:删除 CUDA 语言下 `list(APPEND VLLM_C_LIBS cuda)`,并从 `_C` 扩展的 `define_gpu_extension_target` 参数中移除 `LIBRARIES ${VLLM_C_LIBS}`,CUDA 扩展不再显式链接 cuda 库:

```diff
-if(VLLM_GPU_LANG STREQUAL "CUDA")
-  list(APPEND VLLM_C_LIBS cuda)
-endif()
 define_gpu_extension_target(
   _C
...
-  LIBRARIES ${VLLM_C_LIBS}
   USE_SABI 3
   WITH_SOABI)
```

## 影响与回归风险
- 影响窗口:#12424 合入至本回滚之间构建的 wheel,在无 `libcuda.so` 于 `LD_LIBRARY_PATH` 的环境下受影响;正文未给出具体版本号。
- 回归面:仅改 CMake 链接配置,影响 `_C` CUDA 扩展的构建产物;#12424 原本想解决的 libcuda 链接问题随之回归(作者计划改用 PyTorch 式动态加载方案另解)。

## 关联
- #12505(用户报障 issue)
- #12424(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
