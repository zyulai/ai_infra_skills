# PR #31109: Remove QServe and FBGEMM FP8 quantization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-07-18 |
| 分类 | Quantization(tier: major) |
| Label | documentation, sgl-kernel, run-ci, diffusion, bypass-fastfail, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留部分文件并注明 "8 file(s) dropped" |
| 链接 | https://github.com/sgl-project/sglang/pull/31109 |

## 一句话结论

按 issue #28543 整体移除实验性的 QServe(QoQ)W4A8 与 FBGEMM FP8 量化路径,清理量化方法注册表与 sgl-kernel 内核。

## 动机

正文一句话:依据 issue #28543 移除 experimental 的 QServe (QoQ) W4A8 与 FBGEMM FP8 量化路径。属于代码库瘦身/维护性改动——两条路径长期处于实验状态,移除后减少维护面与构建产物;PR 未声明性能动机或收益。

## 优化点

纯删除,无新机制:(1) Python 侧删除 `layers/quantization/fpgemm_fp8.py`(FBGEMMFp8Config/FBGEMMFp8LinearMethod,含 marlin fp8 回退逻辑)与 `layers/quantization/qoq.py`(QoQConfig),并从 `quantization/__init__.py` 的方法注册表移除 `"qoq"`、`"fbgemm_fp8"` 两项;(2) `model_config.py` 的 `_verify_quantization` 合法量化名列表、`layers/linear.py` 与 `multimodal_gen/runtime/layers/linear.py` 的导出列表中同步删除 `FBGEMMFp8LinearMethod`;(3) sgl-kernel 侧删除 `csrc/gemm/qserve_w4a8_per_chn_gemm.cu`、`benchmark/bench_qserve_w4a8_gemm.py`,并从 `common_extension.cc` 与 `CMakeLists.txt` 摘除对应注册与编译目标。

## 关键改动

```text
python/sglang/srt/layers/quantization/fpgemm_fp8.py: 删除(202 行 FBGEMM FP8 实现)
python/sglang/srt/layers/quantization/qoq.py: 删除(QoQ W4A8 实现)
python/sglang/srt/layers/quantization/__init__.py: 注册表去掉 qoq / fbgemm_fp8
sgl-kernel/csrc/gemm/qserve_w4a8_per_chn_gemm.cu: 删除 W4A8 per-channel GEMM 内核
sgl-kernel/CMakeLists.txt, common_extension.cc: 摘除编译目标与绑定
```

## 性能声明

无。正文与 diff 均未含 benchmark;移除实验路径本身不改变存活路径的内核行为。PR 未注明测试条件。

## 验证与回归风险

依赖 CI(Base 红、Extra 绿)。风险面:仍在使用 `--quantization fbgemm_fp8` 或 `qoq` 的用户启动即失败(配置校验列表已移除);FBGEMM 路径删除后,原先靠 `SGLANG_FORCE_FP8_MARLIN`/marlin fp8 回退的非 FP8 硬件场景需改用其他量化方式;sgl-kernel 的 ABI/扩展导出面变化要求重编 wheel。

## 关联

依据:sglang issue #28543。同区域:同分类 Quantization 其他 PR;存活的可比路径为 fp8(modelopt/压缩张量)与 w4afp8 等(见注册表 diff 上下文)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
