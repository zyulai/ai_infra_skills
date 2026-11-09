# PR #62475: [ROCm][Build] Compile paged attention kernels in parallel (7.6x faster attention build)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-10-21 |
| 分类 | Attention / Kernel (tier: major) |
| Label | rocm, ready, ci/build |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/62475 |

## 一句话结论

把 ROCm paged attention 的模板实例化从单个 3.9k 行翻译单元拆成 12 个 TU 并行编译:attention 编译 547 s→72 s(7.6×),完整 editable 安装 10 min 40 s→1 min 35 s。

## 动机

`csrc/rocm/attention.cu` 是单个 3.9k 行翻译单元,其 kernel 实例化占 ROCm 构建尾部的约 9 分钟(总构建约 10.5 分钟),且完全串行在单核上——HIP 的 TU 由单个 clang 进程编译,没有 nvcc `--threads` 的等价物,多个 `--offload-arch` 也在该进程内串行处理,`MAX_JOBS=64` 无能为力。实例化矩阵被逐轴累积:6 dtype/kv-cache 组合 × head(2) × block(2) × fp8-out(2) × alibi(2) × {NAVI, non-NAVI, interleaved} ≈ 288 个 launcher 实例化、约 860 个 device kernel;近期 #42755 给全部 12 个 kernel 增加 `USE_INTERLEAVED_V_CACHE` 模板参数(+50%)进一步加剧。

## 优化点

机制是纯构建期拆分、零行为变化:`attention.cu` 改名为 `attention_kernels.cuh`(保留 device helpers、kernel/launcher 模板与参数宏,97% 相似度以保住 git blame);新增 `attention_dispatch.cu` 只保留运行时分发(`is_navi_gpu()`、dtype/kv-cache 分发顺序不变),经 `ATTN_HEAD_SWITCH` 转发到固定 head-size 入口;每个 `attention_paged_attn_<head>_<dtype>_<kv>.cu` 约 4 行,包含头文件并以 `ATTN_LAUNCH(...)` 实例化一个组合,共 12 个 TU 由 ninja 并行编译。CMake 在 configure 时校验每个实例化入口都被 dispatch 引用,否则 FATAL_ERROR,避免编译无用文件;新头文件走常规 hipify 流程,无需构建系统特判。

## 关键改动

```text
csrc/rocm/attention_kernels.cuh: 由 attention.cu 改名,kernel/launcher 模板集中于此,新增 ATTN_PARAMS/ATTN_FWD/ATTN_ENTRY/ATTN_LAUNCH 宏
csrc/rocm/attention_dispatch.cu: 新增运行时分发(ATTN_HEAD_SWITCH 12 个入口)
csrc/rocm/attention_paged_attn_*.cu: 12 个单组合实例化 TU(h64/h128 × f16/bf16 × auto/fp8_f16/fp8_fp8)
CMakeLists.txt: VLLM_ROCM_EXT_SRC 列出新文件并做 dispatch 覆盖校验
```

单 TU 内容(全部 12 个同构):

```cpp
#include "attention_kernels.cuh"
ATTN_ENTRY(paged_attention_h128_bf16_auto) {
  ATTN_LAUNCH(__hip_bfloat16, __hip_bfloat16,
              vllm::Fp8KVCacheDataType::kAuto, 128, MFMAType::F16);
}
```

## 性能声明

测试条件:MI350,`vllm/vllm-openai-rocm:nightly-rocm100-39983a5e...`,`PYTORCH_ROCM_ARCH=gfx950`,默认 `RelWithDebInfo`(-O2 -g),计时来自冷构建 `.ninja_log`(`CCACHE_DISABLE=1 ninja -j64 _rocm_C`):

- attention kernel 编译(墙钟):main 547 s(单核)→ 本 PR 72 s(12 核,各 TU 58–72 s),即 7.6×;`attention_dispatch.cu` 15 s。
- `_rocm_C.abi3.so` 链接 ~2 s → 0.26 s;完整 `MAX_JOBS=64 pip install -e .` 10 min 40 s → 1 min 35 s。
- 代价:`.so` 从 196.3 MB 增至 219.2 MB(+11.6%,增量是各 TU 重复的 device DWARF 调试信息,非代码),总 CPU 工作量约 +30%;`-g1`/`-g0` 分别可到 96.1/74.9 MB,留作后续 PR。
- 佐证:head=64 限制(一半实例化)270 s vs 547 s,说明编译时间与实例化数线性,拆分可得近似 N× 墙钟收益。

## 验证与回归风险

内核正确性矩阵覆盖 12 个 TU:head 64/128 × block 16/32 × fp16/bf16 × kv auto/fp8 × alibi on/off × interleaved-V on/off × GQA 1/4/8——f16 MFMA 路径 128 passed(414 s)、`VLLM_ROCM_FP8_MFMA_PAGE_ATTN=1` fp8 路径 64 passed(206 s)、head-size default 分支 7 passed。无模型评测(diff 不改变 kernel、dispatch 与数值,只改变编译成多少个 object)。遗留缺口(P 明确列出):block-size default 分支、CI 中 fp8_fp8 TU、`fp8_out_scale`(OUTT=uint8)、`fp8_e4m3` 别名、NAVI 区域在 `gfx1100;gfx1201` 下的编译均无自动化覆盖。新增组合需同时加 TU 与 CMake 条目。

## 关联

实例化矩阵相关前序:#8310、#8577、#12348、#17139、#17004、#22222、#42755;正交的构建优化:#17419(构建慢 RFC)、#60906(editable install 复用 CMake dir);同区域内核卡:pr-12348(faster custom paged attention)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
