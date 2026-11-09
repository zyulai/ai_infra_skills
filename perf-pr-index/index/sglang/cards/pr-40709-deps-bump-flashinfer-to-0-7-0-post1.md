# PR #40709: [Deps] Bump FlashInfer to 0.7.0.post1

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-09-30 |
| 分类 | Attention / Kernel |
| Label | documentation, dependencies, run-ci, diffusion, jit-kernel, run-ci-extra, release-highlight, bypass-fail-fast, highest-priority |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 10 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/40709 |

## 一句话结论

把 FlashInfer 依赖从 0.6.18 升级到 0.7.0.post1(连带 cutlass-dsl 4.8.0、quack-kernels 0.6.5),适配新版本 plan 校验与 per-arch jit-cache 布局,为所有 FlashInfer 注意力路径带来新内核版本。

## 动机

PR 正文为空(仅 CI 状态),动机需从 diff 推断:统一升级 FlashInfer 依赖版本。0.7.0 起有两项破坏性变化需要适配——plan 阶段拒绝 `kv_len_arr` 与 `kv_indptr` 不一致的输入(影响 draft 分支的 MLA decode plan);`flashinfer-jit-cache` 从单 wheel 改为 per-arch dist(经 entry points 发现,shim 无 provider 时仍可导入并报告正确版本,静默降级)。此结论来自代码变更本身,PR 正文未陈述。

## 优化点

版本提升:pyproject `flashinfer_python[cu13]` 0.6.18 -> 0.7.0.post1,`nvidia-cutlass-dsl` 4.6.2 -> 4.8.0,`quack-kernels` 0.6.4 -> 0.6.5;Dockerfile / Dockerfile.cu134 的 `FLASHINFER_VERSION` 同步,cu134 移除 `CUTLASS_DSL_VERSION` ARG;engine 启动时 `assert_pkg_version` 检查同步到 0.7.0.post1。运行时适配:flashinfer_mla_backend 的非 replay plan 路径把 `kv_lens` 重算为 `kv_indptr` 差分(draft 分支 span = prefix + 每 step 一个,绕过新 plan 校验;注释指明经 `fast_mla_decode_plan` 跳过该检查);cute FA 工具里 `quack.activation.sub_packed_f32x2` 改为 `cute.arch.sub_packed_f32x2`(移除 quack.activation import)。CI/构建适配:jit-cache 下载脚本从单 wheel 匹配改为 glob 整组 per-arch wheel 并 `--no-index` 安装,新增 `ci_check_flashinfer_jit_cache.sh`(校验 `get_jit_cache_providers()` 非 空,防止 shim 无 provider 的静默降级),`uninstall_stale_flashinfer` 按实际安装枚举并卸载 `flashinfer-jit-cache-<arch>` 包。

## 关键改动

```text
python/pyproject.toml: flashinfer_python[cu13]==0.7.0.post1、cutlass-dsl 4.8.0、quack-kernels 0.6.5
python/sglang/srt/layers/attention/flashinfer_mla_backend.py: 非 replay plan 的 kv_lens 重算以通过 0.7.0 plan 校验
python/sglang/kernels/ops/attention/flash_attn/cute/utils.py: sub_packed_f32x2 从 quack.activation 迁到 cute.arch
scripts/ci/cuda/ci_check_flashinfer_jit_cache.sh: 新增 per-arch provider 有效性校验
scripts/ci/cuda/ci_install_dependency.sh: 枚举并卸载 stale per-arch jit-cache 包
```

## 性能声明

PR 未注明任何 benchmark 数字。升级本身使 SM100/Blackwell 相关 FlashInfer 内核进入 0.7.0.post1 版本(如 KDA `recurrent_kda`、trtllm MLA decode 等使用方),但 PR 未提供前后性能对比;使用方 PR 各自报告数字(如 PR #43735 注明 FlashInfer 0.7.0.post1 下 B300 有 64-head kernel bug)。

## 验证与回归风险

CI 状态:Extra 通过,Base 失败(x),AMD ROCm 10 沙漏(labels 含 bypass-fail-fast)。风险面:0.7.0 plan 校验变严,本 PR 用 kv_indptr 差分绕过(draft 分支),其他调用点若存在 kv_len 与 indptr 不一致的场景仍可能触发;per-arch jit-cache 若 provider 缺失会静默无 cubin(新增校验脚本兜底,但校验失败即 exit 1 重装);cutlass-dsl 4.6.2 -> 4.8.0 与 quack-kernels 0.6.4 -> 0.6.5 属连带升级, cute FA 路径行为需回归;SM120 Sage backend 所需 API(FlashInfer #4691)仍不在该 pin 中,subblock_sparse README 更新为需另行装源码版本。

## 关联

同区域:所有 FlashInfer 后端使用方——PR #30113(KDA recurrent_kda)、PR #43735(DSA DCP trtllm sparse attention,注明 0.7.0.post1 的 64-head bug)、PR #29458 时代的 prefill CUDA graph flashinfer wrapper。依赖链:cutlass-dsl 4.8.0、quack-kernels 0.6.5、flashinfer-jit-cache per-arch 布局。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
