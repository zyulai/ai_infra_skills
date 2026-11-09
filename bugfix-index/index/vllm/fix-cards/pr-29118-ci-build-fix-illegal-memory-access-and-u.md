# PR #29118: [CI/Build] Fix illegal memory access and unsupported test in kernels/attention/test_cache.py

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29118](https://github.com/vllm-project/vllm/pull/29118) |
| 作者 | rasmith |
| 合并日期 | 2025-11-21 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
test_cache.py 的 Triton kernel 测试只设 default_device 未设当前设备导致 IMA;补齐并跳过 ROCm 不支持的 fp8_ds_mla。

## 问题现象
`tests/kernels/attention/test_cache.py` 中测试 Triton kernel 的用例出现 illegal memory access。触发条件为正文所述:凡是走 Triton kernel 的测试,需要同时调用 `torch.set_default_device` 和 `torch.cuda.set_device` 且指向同一设备,才能保证 kernel 执行在已分配张量所在设备上;此前测试只调用了前者。另外 `test_concat_and_cache_ds_mla` 在 ROCm 上因 `concat_and_cache_mla` 不支持 `fp8_ds_mla` 而出错,属不支持的平台组合被当作失败。影响面为 AMD 平台上的该测试文件。

## 根因
正文披露的根因:Triton kernel 的执行设备取决于当前 CUDA 设备,仅 `torch.set_default_device` 只决定张量分配位置,不改变当前设备,两者不一致时 kernel 便在与张量不同的设备上执行,产生非法内存访问。第二个问题是平台能力问题:`fp8_ds_mla` 在 ROCm 上不被 `concat_and_cache_mla` 支持,需要显式跳过而非让其报错。

## 修复方案
只改 `tests/kernels/attention/test_cache.py`:
- 给 `test_copy_blocks`、`test_reshape_and_cache`、`test_reshape_and_cache_flash`、`test_concat_and_cache_mla`、`test_concat_and_cache_ds_mla`、`test_copy_blocks_mla`、`test_swap_blocks_mla` 共 7 个用例,在 `torch.set_default_device(device)` 后补一行:

```diff
     torch.set_default_device(device)
+    torch.cuda.set_device(device)
```

- `test_concat_and_cache_ds_mla` 开头增加 ROCm 跳过:

```diff
+    if current_platform.is_rocm():
+        pytest.skip("concat_and_cache_mla doesn't support fp8_ds_mla on ROCm")
```

正文给出修复后 AMD 测试结果:`1110 passed, 294 skipped, 3 warnings in 741.42s (0:12:21)`。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:纯测试代码改动,不影响任何生产路径。仅影响 `test_cache.py` 的执行:设备一致性修正在 CUDA/ROCm 上同样生效;ROCm 上 `test_concat_and_cache_ds_mla` 由失败转为 skip(跳过计数增加)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
