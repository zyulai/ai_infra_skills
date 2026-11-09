# PR #48845: [ROCm][CI] Fix AITER MLA fp8 decode metadata regression test

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48845](https://github.com/vllm-project/vllm/pull/48845) |
| 作者 | stefankoncarevic |
| 合并日期 | 2026-07-21 |
| labels | rocm, ready |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#47276 把 gfx950 DPA+FP8 KV 的 decode query dtype 改为 fp8 后,旧 golden 测试未同步而失败;仅更新测试期望值修复。

## 问题现象
gfx950 上的 AITER MLA persistent-decode 回归测试 `tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py` 在 main 上开始失败。失败断言为:AITER MLA persistent decode metadata 与 fp8/bf16 golden 在 `work_indptr`、`work_info_set`、`reduce_indptr`、`reduce_final_map`、`reduce_partial_map` 五个字段上不匹配,提示 builder 必须转发 dtype_q/dtype_kv 给 `get_mla_metadata_v1`。修复后该测试通过(1 passed, 16 warnings in 6.52s,MI355X/gfx950 主机)。

## 根因
正文披露:两个 bugfix 相互碰撞——#46997 添加该测试并把 `dtype_q`/`dtype_kv` 转发给 `aiter.get_mla_metadata_v1`,当时 decode query dtype 还是 bf16,故测试固化 `EXPECTED_Q_DTYPE = torch.bfloat16`;其后 #47276("Fix memory access fault in AITER MLA backend for DPA+FP8 KV")为避免内存访问错误,有意把 gfx950 DPA+FP8 KV 路径的 decode query dtype 改为 fp8——这是正确的运行时行为,但 #46997 的测试未随之更新,golden 仍按 dtype_q=bf16 重算 split/reduce 元数据,布局自然与 fp8 前向不一致。

## 修复方案
仅改测试,不动生产代码:把 `EXPECTED_Q_DTYPE` 与 `EXPECTED_KV_DTYPE` 从硬编码的 `torch.bfloat16` / `torch.float8_e4m3fn` 改为 `current_platform.fp8_dtype()`,并在注释中引用 #47276 说明 fp8 KV-cache 路径下 builder 会转发平台 fp8 dtype:

```diff
-EXPECTED_Q_DTYPE = torch.bfloat16
-EXPECTED_KV_DTYPE = torch.float8_e4m3fn
+EXPECTED_Q_DTYPE = current_platform.fp8_dtype()
+EXPECTED_KV_DTYPE = current_platform.fp8_dtype()
```

## 影响与回归风险
- 影响窗口:#47276 合并之后至本 PR 之前,gfx950 ROCm CI 上的该回归测试一直误报。
- 回归面:仅 `tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py` 的期望值与注释;生产代码(builder、kernel)零改动。期望值改用 `current_platform.fp8_dtype()` 而非硬编码字面量,对平台 fp8 dtype 与 torch 字面量不一致的场景更稳健。

## 关联
- PR #46997(添加测试并转发 dtype)
- PR #47276(将 decode query dtype 改为 fp8 的 bugfix)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
