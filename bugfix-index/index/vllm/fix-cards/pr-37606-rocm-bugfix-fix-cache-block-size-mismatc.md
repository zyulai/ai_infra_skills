# PR #37606: [ROCm][Bugfix] fix cache block size mismatch for aiter unified attention

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37606](https://github.com/vllm-project/vllm/pull/37606) |
| 作者 | divakar-amd |
| 合并日期 | 2026-03-20 |
| labels | bug, ready, rocm, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
AITER unified attention 用 `--attention-config` 指定后端时 block size 仍为 16,改为由后端 `get_preferred_block_size` 统一返回 64。

## 问题现象
使用环境变量 `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` 时,cache block size 能按预期解析为 64;但当改用 `--attention-config '{"backend": "ROCM_AITER_UNIFIED_ATTN"}'` 指定同一后端时,cache block size 仍解析为 16,与 AITER unified attention 期望的 64 不匹配(Resolves issue #37548)。影响所有通过参数方式选择 ROCM_AITER_UNIFIED_ATTN 后端的 ROCm 用户。

## 根因
PR 正文披露:block size 设置逻辑放在 ROCm 平台的 `check_and_update_config` 里,且只判断环境变量路径(`envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`),没有覆盖经 `--attention-config` 参数选择后端的方式,导致同一后端因配置方式不同而得到不同 block size。正文指出这与 PR #35122 对 `check_and_update_config` 的近期重构不一致。

## 修复方案
- `vllm/platforms/rocm.py`:删除 `check_and_update_config` 中基于环境变量写死 `cache_config.block_size`(64/16)的分支,并移除空的 `update_block_size_for_backend` 存根;
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`:在后端类上新增 `get_preferred_block_size`,无论后端通过环境变量还是参数选中,都统一返回 64:

```diff
+    @classmethod
+    def get_preferred_block_size(cls, default_block_size: int) -> int:
+        logger.warning_once(
+            "[ROCM_AITER_UNIFIED_ATTN]: Setting kv cache block size to 64."
+        )
+        return 64
```

其余后端(`ROCM_AITER_FA`、`ROCM_ATTN`、`TRITON_ATTN`)继续默认 cache block size = 16。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:ROCm 平台不再在 `check_and_update_config` 中直接改写 `cache_config.block_size`,该值改由注意力后端的 `get_preferred_block_size` 提供;ROCM_AITER_UNIFIED_ATTN 用户的 block size 从 16 变为 64,会改变 KV cache 分块与显存占用;其余 ROCm 后端行为保持不变。

## 关联
- Resolves https://github.com/vllm-project/vllm/issues/37548
- 参考 https://github.com/vllm-project/vllm/pull/35122

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
