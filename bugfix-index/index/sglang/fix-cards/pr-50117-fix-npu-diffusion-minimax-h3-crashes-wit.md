# PR #50117: [Fix] [NPU] diffusion: MiniMax-H3 crashes with mxfp8 quantization or ref2va configuration

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#50117](https://github.com/sgl-project/sglang/pull/50117) |
| 作者 | Napkin-AI |
| 合并日期 | 2026-11-17 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论

MiniMax-H3 在 NPU MXFP8 量化下崩溃,因为它把 Diffusion 专有的 `accepts_mxfp8_input` 方法调用到 SRT 量化方法对象上;另外 ref2va 参考视频解码在 `os.memfd_create` API 不存在的 Python 构建里抛 `AttributeError`(原代码只捕获 `OSError`)。修复:能力查询前先判 `isinstance(linear.quant_method, LinearMethodBase)`,并把 memfd 创建集中到带 API 可用性检查的辅助函数。

## 问题现象

两个独立崩溃(正文 Motivation):

1. MiniMax H3 使用 NPU MXFP8 时崩溃:Diffusion 侧代码在 `_accepts_mxfp8_input` 中对 linear 层的 `quant_method` 无条件调用 `accepts_mxfp8_input(linear)`,但 SRT/NPU 的量化方法对象没有该方法,调用即崩。
2. ref2va 配置下参考视频解码失败:worker 通过 `os.memfd_create` 让 ffmpeg 把 RGB 流写入匿名文件描述符;当 Python 构建中根本没有 `os.memfd_create` API 时,抛出的是 `AttributeError`,而现有 handler 只捕获 `OSError`,异常逃逸导致解码失败。

## 根因

正文披露:一是量化方法能力查询缺少类型守卫——`accepts_mxfp8_input` 是 Diffusion 层 `LinearMethodBase` 的接口,并非所有 `LinearBase.quant_method` 实现都有;二是 memfd 使用点直接引用 `os.memfd_create` 属性,只处理了 `OSError`(syscall 被容器 seccomp 屏蔽的情况),未处理 API 本身不存在的情况。

## 修复方案

1. `python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`:
   - `_accepts_mxfp8_input` 在调用 `accepts_mxfp8_input` 前增加 `isinstance(linear.quant_method, LinearMethodBase)` 判定,SRT/NPU 方法保持标准张量输入路径。
   - `vsa_h3_fold_gate` 的导入从模块顶层移到 `fold_after_exchange` 分支内延迟导入,消除不必要的顶层依赖。
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py`:新增 `_create_reference_video_memfd`,集中处理——非 Linux 或没有 `os.memfd_create` 属性直接返回 -1;`OSError`(API 存在但 syscall 不可用或被 seccomp 阻止)也返回 -1;`_decode_reference_video_local` 与 `_decode_reference_video_shared`(leader 分支,`leader_fd < 0` 时回落到本地解码路径)都改用它。

关键 diff 摘录:

```diff
     return (
         isinstance(linear, LinearBase)
         and linear.quant_method is not None
+        and isinstance(linear.quant_method, LinearMethodBase)
         and linear.quant_method.accepts_mxfp8_input(linear)
     )
```

```diff
+def _create_reference_video_memfd(name: str) -> int:
+    """Return an anonymous output fd, or -1 when memfd is unavailable."""
+    if not sys.platform.startswith("linux") or not hasattr(os, "memfd_create"):
+        return -1
+    try:
+        return os.memfd_create(name, flags=os.MFD_CLOEXEC)
+    except OSError:
+        # The Python API can exist while the syscall is unavailable or blocked
+        # by the container's seccomp policy.
+        return -1
```

## 影响与回归风险

- 影响窗口:multimodal_gen diffusion 管线中 MiniMax-H3 的 MXFP8 量化路径与 ref2va 参考视频编码路径;SRT/NPU 量化方法在能力探测时不再被误调 Diffusion 接口。
- 准确性:正文说明"本 PR 不影响 accuracy,mxfp8 的视频质量会略有不同",附 BF16 与 MXFP8 生成视频对照链接;`fold_after_exchange` 分支行为不变(仅导入方式调整)。
- 性能数据(Ascend950PR_957c、CANN 9.1.1、torch/torch_npu 2.10.0,8 卡、TP2 + Ulysses 4、CacheDit 开启、51 步、seed 1101):E2E latency BF16 298993.77 ms → MXFP8 287176.09 ms(-4.0%)/ MXFP8 Transformer 286525.90 ms(-4.2%);allocated_mb 45,934.55 → 30,673.38(-33.22%);peak_allocated_mb 51,718.00 → 36,456.83(-29.51%);peak_reserved_mb 61,252.00 → 53,102.00(-13.31%)。
- 回归面:memfd 回退路径在本 PR 前后语义一致(-1 时走非 memfd 解码),新增的仅是 API 缺失时的回退;Checklist 各项未勾选,无新增单测。CI:Base 通过,Extra 与 AMD 失败(失败原因 pack 未提供)。

## 关联

- 涉及文件:`python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`、`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py`、`python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn_h3.py`(`vsa_h3_fold_gate`,改为延迟导入)。
- 复现命令与 `h3-ref2va.json` 配置见 PR 正文;测试环境 Ascend950PR_957c。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
