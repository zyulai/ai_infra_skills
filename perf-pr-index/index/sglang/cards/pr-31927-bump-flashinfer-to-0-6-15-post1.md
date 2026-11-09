# PR #31927: Bump FlashInfer to 0.6.15.post1

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-07-22 |
| 分类 | Attention / Kernel(tier: major) |
| Label | dependencies, deepseek, blackwell, run-ci, bypass-fastfail, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 全量保留(约 10 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/31927 |

## 一句话结论

升级 FlashInfer 到 0.6.15.post1,修掉 MLA decode 与 fused-MoE 的 host 端开销回归,恢复长上下文吞吐。

## 动机

上一轮 0.6.15 bump(#31502)曾因 host 端回归拉低长上下文 serving 吞吐而被 #31625 revert。0.6.15.post1 包含两个上游修复,正好对症:一是 MLA decode `TuningConfig` 对象未做记忆化,等价输入反复生成不相等的 initializer closure,在 `_find_nearest_profile` 中形成 hash-collision 链,使单次调用 host 开销 O(n)、长期运行累计 O(n²);二是 fused-MoE runner 每次调用都重新构造并全量校验 tactic,带来稳定的 host dispatch 开销。此外 0.6.15 的兼容性改动(如 CuteDSL `ActivationType` API)需要保留。

## 优化点

核心是依赖版本升级携带的两个上游修复(flashinfer#3970 与 #4045):前者把 MLA decode 的 `TuningConfig` 按等价输入记忆化,消除长时 MLA decode 的 O(n²) host 开销并改进 autotuner-cache 校验/自愈;后者按 host thread 缓存构造期派生的 fused-MoE runner、只校验选中的 tactic pair,每次调用省约 0.4 ms 的 warmed host-dispatch 开销。SGLang 侧配套:为 trtllm-gen MLA decode 常驻 `multi_ctas_kv_counter_buffer`(按 max_running_requests 预分配、必要时增长)避免每次分配;删除 DeepSeek `modelopt_fp4 + flashinfer_trtllm` 下 correction_bias 强制 bf16 的过时 workaround;CuteDSL MoE wrapper 迁移到显式 `ActivationType` 枚举。

## 关键改动

```text
python/pyproject.toml, docker/Dockerfile, engine.py: flashinfer_python/cubin/JIT-cache pin 0.6.14 → 0.6.15.post1
python/sglang/srt/layers/attention/trtllm_mla_backend.py: 常驻 multi_ctas_kv_counter_buffer + grow_if_needed
python/sglang/srt/layers/attention/dsa_backend.py: DSA trtllm 路径接入同一 counter buffer
python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py: activation → ActivationType 枚举映射
python/sglang/srt/models/deepseek_v2.py: 删除 modelopt_fp4+trtllm 的 bf16 correction_bias workaround
```

## 性能声明

PR 正文给出(flashinfer 上游侧,未附独立复现数据):

- flashinfer#4045:每次调用移除约 0.4 ms 的 warmed host-dispatch 开销;端到端 GB200/GB300("GB300")测试恢复了对 compatibility baseline 约 6% 的吞吐差距。
- flashinfer#3970:消除长时 MLA decode workload 的 per-call O(n) / 总计 O(n²) host 开销(未给绝对数字)。
- 佐证:GLM-5.2 NVFP4 TP+MTP e2e 测试的 bs=1 速度阈值从 250 提到 280(`test_dsa_glm52_nvfp4_tp_mtp.py`),未注明单位与硬件(硬件应为 Blackwell 系,依据 label,正文未明说)。

## 验证与回归风险

PR 列出的验证均为静态/离线检查(pre-commit、TOML 解析、PEP 440 版本序、Docker/依赖 pin 一致性检查等),并注明 "GPU model E2E tests were not run locally";性能恢复依赖 flashinfer 上游的 GB300 测试结论。风险点:所有走 flashinfer MLA/MoE 路径(DeepSeek、GLM、Blackwell)的行为随 0.6.15.post1 变化;`multi_ctas_kv_counter_buffer` 按批大小增长逻辑在超大 batch 时会重新分配;CI 状态显示 Latest PR Test 未通过(红叉),合并时点风险未知。

## 关联

前序:#31502(原始 0.6.15 bump)、#31625(其 revert)。上游:flashinfer#3970、flashinfer#4045。同区域:其他 FlashInfer bump 类 PR(#48468 bump 到 0.7.3)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
