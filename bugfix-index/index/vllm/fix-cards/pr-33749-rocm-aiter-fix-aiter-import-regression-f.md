# PR #33749: [ROCm][AITER] Fix AITER import regression for explicit backend selection

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33749](https://github.com/vllm-project/vllm/pull/33749) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-02-06 |
| labels | rocm, speculative-decoding, ready, v1 |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#32902 引入回归,ROCm 上无法显式选择 AITER 注意力后端;将「可用性」与「环境变量偏好」分离并补齐缺失导入修复。

## 问题现象
在 ROCm 上,当 `VLLM_ROCM_USE_AITER=0`(或未设置)时,用户无法通过 `attention_config={"backend": "ROCM_AITER_FA"}` 显式选择 AITER 后端,即使该后端实际可用。报错为 `AttributeError: 'builtin_function_or_method' object has no attribute 'flash_attn_varlen_func'`。此外,`fa_utils.py` 缺失 ROCm 专属导入,导致 Eagle 相关测试在 ROCm 上因 import 错误而失败;测试套件中还存在一个调用已被删除 custom op 的过时测试。

## 根因
正文披露:回归由 #32902 引入。为避免 JIT 编译告警,原实现把 `VLLM_ROCM_USE_AITER` 环境变量检查错误地放进了 `is_aiter_found_and_supported()` 的可用性判断,使显式选择路径也被 env var 拦截;Eagle 的 `allowed_attn_types` 校验在 env var 未设置时拒绝 `AiterFlashAttentionMetadata`。正文给出明确设计区分:自动发现路径应尊重 env var(避免 JIT 告警),显式选择路径则应无视 env var,需实现 OR 逻辑(env var=1 或显式选择任一成立即可导入 aiter)。

## 修复方案
1. `vllm/_aiter_ops.py`:`is_aiter_found_and_supported()` 移除 env var 检查,只检查平台+库+架构(可用性),env var 检查移回模块级导入处以抑制 JIT 告警;新增不带 `@is_aiter_supported` 装饰器的 `rocm_aiter_ops.flash_attn_varlen_func` / `pa_fwd_asm` 懒导入包装。关键改动:

```diff
 def is_aiter_found_and_supported() -> bool:
-    if current_platform.is_rocm() and IS_AITER_FOUND and envs.VLLM_ROCM_USE_AITER:
+    if current_platform.is_rocm() and IS_AITER_FOUND:
```

2. `vllm/v1/spec_decode/eagle.py`:`allowed_attn_types` 改为仅用 `find_spec` 检查后端模块是否存在(能力检查),不再叠加 `is_enabled()` 条件。
3. `vllm/v1/attention/backends/fa_utils.py`:补齐 ROCm 分支缺失的导入——`flash_attn_varlen_func`、`get_scheduler_metadata` 桩(ROCm 不用 FA3 的 scheduler metadata)、`reshape_and_cache_flash`(取自 C++ custom op);`is_flash_attn_varlen_func_available()` 在 ROCm 上改为检查模块级标志 `_ROCM_FLASH_ATTN_AVAILABLE`(修复此前恒返回 False 的 bug)。
4. `vllm/v1/attention/backends/rocm_aiter_fa.py`:所有 `aiter.flash_attn_varlen_func` / `aiter.pa_fwd_asm` 调用点改走 `rocm_aiter_ops` 包装,并加注释说明模块级无条件 import aiter 的设计意图。
5. `tests/kernels/attention/test_aiter_flash_attn.py`:把调用已删除 custom op `torch.ops.vllm.flash_attn_varlen_func` 的过时测试迁移为现行两步架构(`cp_mha_gather_cache` Triton kernel 取 KV + `aiter.flash_attn_varlen_func` 计算)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本号;回归自 #32902 引入起,至本修复合并前。
- 回归面:覆盖 `_aiter_ops.py` 的 FP8 dtype 常量(`AITER_FP8_DTYPE`→`FP8_DTYPE`)与多个量化 op 断言、ROCm AITER 注意力后端全部调用点、Eagle spec decode 的后端类型校验、`fa_utils.py` 平台分支;diff 中 `is_shuffle_kv_cache_enabled` 由 `_AITER_ENABLED and _SHUFFLE_KV_CACHE_ENABLED` 简化为仅 `_SHUFFLE_KV_CACHE_ENABLED`,可能影响 KV cache shuffle 判定路径(由 diff 推断);显式选择路径在 env var 关闭时现在会真正触发 aiter 导入,需注意 JIT 告警回归的可能。

## 关联
- #32902(正文指明的回归引入 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
