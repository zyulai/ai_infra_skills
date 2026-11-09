# PR #32837: [Hardware][AMD][CI][Bugfix] Fix regressions from deprecated env vars

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32837](https://github.com/vllm-project/vllm/pull/32837) |
| 作者 | mawong-amd |
| 合并日期 | 2026-01-22 |
| labels | bug, ci/build, kv-connector, ready, rocm, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
环境变量废弃(#32812)引发两处 ROCm 回归:vllm config 未就绪时取值报错、ROCM_ATTN 传不进 NIXL 测试;分别以 or-none 判空和 env 传参修复。

## 问题现象
AMD ROCm 上 #32812 废弃环境变量后出现测试回归,AMD CI nightly(build 3452)三组失败:1) V1 Test attention (H100);2) NixlConnector PD accuracy tests (Distributed);3) DP EP NixlConnector PD accuracy tests (Distributed)。具体表现:attention 后端选择测试 `v1/attention/test_rocm_attention_backends_selection.py::test_standard_attention_backend_selection` 回归,以及 NIXL 精度测试没拿到要求的 `ROCM_ATTN` 后端而失败。

## 根因
正文披露两点:
1. `VLLM_V1_USE_PREFILL_DECODE_ATTENTION` 被废弃后,`RocmPlatform.get_attn_backend_cls` 改为调用 `get_current_vllm_config` 判断是否用 `ROCM_ATTN` 后端;但存在 current vLLM config 尚未设置的时机,该函数此时会报错。
2. `VLLM_ATTENTION_BACKEND` 被废弃后,测试侧所需的 `ROCM_ATTN` 后端不再被正确传入 NIXL 精度测试,导致测试失败。

## 修复方案
- `vllm/platforms/rocm.py`:`get_attn_backend_cls` 改用 `get_current_vllm_config_or_none()`,config 为 None 时跳过判断,不再报错。
- `.buildkite/test-amd.yaml`:NIXL 两条测试命令改为 `ROCM_ATTN=1`/`DP_EP=1 ROCM_ATTN=1` 环境变量方式传后端选择。
- `tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh`:读取 `ROCM_ATTN` 环境变量,非空时以 `--attention-backend ROCM_ATTN` 运行:

```diff
-run_tests "default backend" ""
+if [[ -n "${ROCM_ATTN:-}" ]]; then
+  echo "ROCM_ATTN is set, running with --attention-backend ROCM_ATTN"
+  run_tests "ROCM_ATTN backend" "--attention-backend ROCM_ATTN"
+else
+  run_tests "default backend" ""
+fi
```

三组测试在修复后均通过。

## 影响与回归风险
- 影响窗口:#32812(环境变量废弃)合入至本 PR 之间的 ROCm CI 受影响;正文未给出具体版本号。
- 回归面:`RocmPlatform.get_attn_backend_cls` 的后端选择路径(config 未就绪时行为改变,不再抛错而是继续走后续优先级);AMD CI 的 NIXL 精度测试脚本与流水线配置;不涉及核心推理逻辑。

## 关联
- #32812(引入环境变量废弃、造成回归的 PR)
- AMD CI nightly build 3452(正文链接的失败构建)
- 正文未列出其他 issue/PR

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
