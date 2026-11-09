# PR #20303: bump fa4 to official released fa4 pkg

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @rainj-me |
| 合并 | 2026-03-18 |
| 分类 | Attention / Kernel(classify.py 给出,tier: major) |
| Label | dependencies, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/20303 |

## 一句话结论

把 FlashAttention-4 集成从私有 `sgl-fa4` 包切换到官方发布的 `flash-attn-4` 包,并同步升级 cutlass-dsl 与 quack-kernels 依赖。

## 动机

此前 SGLang 依赖内部包 `sgl-fa4==4.0.5` 提供 FA4 的 CUTE 接口;官方 FA4 包发布后应切换到公开发行的 `flash-attn-4`,减少内部 fork 维护成本并跟随上游版本演进。PR 正文 Motivation 仅一句 "fa4 integration with official fa4 pkg",未展开更多背景,本卡如实不引申。

## 优化点

依赖切换而非算法改动:`pyproject.toml` 中 `sgl-fa4==4.0.5` 替换为官方 `flash-attn-4>=4.0.0b4`;`python/sglang/jit_kernel/flash_attention_v4.py` 的导入从 `sgl_fa4.cute.flash_attn_varlen_func` 改为 `flash_attn.cute.flash_attn_varlen_func`(错误信息同步更新)。同时升级配套依赖:`nvidia-cutlass-dsl` 4.3.5 pin 放宽为 `>=4.4.1`,`quack-kernels` 0.2.4 → `>=0.3.0`,`cuda-python` 在 CUDA13 分支升到 13.2.0(CUDA12 分支补装 12.9)。Dockerfile 与 CI 安装脚本(`ci_install_dependency.sh`)同步:移除对 `nvidia-cutlass-dsl-libs-base` 的清理和旧版强制重装逻辑、删除先前为避免冲突手动删除 `flash_attn` 目录的 workaround、卸载清单加入 `sgl-fa4 flash-attn-4`。

## 关键改动

```text
python/pyproject.toml: sgl-fa4==4.0.5 -> flash-attn-4>=4.0.0b4;cutlass-dsl/quack-kernels 升级
python/sglang/jit_kernel/flash_attention_v4.py: FA4 varlen 入口改从官方 flash_attn.cute 导入
docker/Dockerfile: cutlass-dsl >=4.4.1 安装方式与 CUDA12/13 分支包调整
scripts/ci/cuda/ci_install_dependency.sh: 卸载旧 sgl-fa4、移除 flash_attn 目录 workaround
```

```python
try:
    from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception as _e:  # pragma: no cover
    _flash_attn_varlen_func = None
    _flash_attn_import_error = _e
```

## 性能声明

PR 正文未提供 benchmark 或性能对比数据(Benchmarking 章节为空模板),未注明测试条件。

## 验证与回归风险

PR 正文未列新增测试;以 CI(run-ci label)验证。风险点:FA4 调用路径(`jit_kernel/flash_attention_v4.py`)的导入包名变化,环境若残留旧 `sgl-fa4` 会与之冲突,CI 脚本已将其加入卸载清单;`nvidia-cutlass-dsl` 从精确 pin 放宽为 `>=4.4.1` 后,cutedsl 相关 kernel(含 FlashInfer CuTe DSL、quack-kernels)行为随上游版本浮动;Dockerfile 中 `nvidia-cublas`、`nixl-cu13` 等 CUDA13 分支包同步变化。

## 关联

同区域:FlashInfer/CuTe DSL 依赖类 PR(PR #36954、#43494、#44591 的 FlashInfer bump 属相邻依赖链);后续 FA4 pin 演进见 `flash-attn-4>=4.0.0b18`(PR #44591 时的 pyproject)。PR 正文未列出其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
