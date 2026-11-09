# PR #41113: [Bugfix] Fix rope

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41113](https://github.com/vllm-project/vllm/pull/41113) |
| 作者 | jeejeelee |
| 合并日期 | 2026-04-29 |
| labels | bug, ready |
| 热度 | 👍 5 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
rope CUDA kernel 假定 cos/sin cache 为 fp32;新增 cache_t 模板按 cache 实际 dtype 分发,去除强制转换。

## 问题现象
正文未描述具体现象,Purpose 一栏仅写 "FIX https://github.com/vllm-project/vllm/issues/41017";Test Plan 与 Test Result 均为空。

## 根因
正文未披露根因。(由 diff 推断)旧代码在 host 端将 cos_sin_cache 强制 `cos_sin_cache.to(torch::kFloat32)` 后以 `const float*` 传入 kernel,kernel 各处假定 cache 为 fp32;新代码引入 `cache_t` 模板参数并按 `cos_sin_cache.scalar_type()` 分发,cos/sin 读取处改为 `static_cast<float>`,暗示问题与 cache dtype 的处理(强制转换/精度)有关。

## 修复方案
`csrc/pos_encoding_kernels.cu`:
- `apply_token_rotary_embedding`、`apply_rotary_embedding`、`rotary_embedding_kernel` 均新增 `cache_t` 模板参数,cos/sin/cache 指针由 `const float*` 改为 `const cache_t*`,加载时经 `static_cast<float>`;
- host 端 `rotary_embedding` 删除 `auto cache_f32 = cos_sin_cache.to(torch::kFloat32);`,改为在 query dtype 分发内再按 `cos_sin_cache.scalar_type()` 做 `VLLM_DISPATCH_FLOATING_TYPES`,以 cache 原始 dtype 指针直接启动 kernel;
- `tests/kernels/core/test_rotary_embedding.py` 为 `test_rotary_embedding_opcheck` 新增 `dtype` 参数化(`torch.float32, torch.bfloat16`),RotaryEmbedding 与 query/key 均按该 dtype 构造。

```cpp
-    cos_f = VLLM_LDG(cos_ptr + x_index);
-    sin_f = VLLM_LDG(sin_ptr + x_index);
+    cos_f = static_cast<float>(VLLM_LDG(cos_ptr + x_index));
+    sin_f = static_cast<float>(VLLM_LDG(sin_ptr + x_index));
```

```cpp
-  auto cache_f32 = cos_sin_cache.to(torch::kFloat32);
+    VLLM_DISPATCH_FLOATING_TYPES(
+        cos_sin_cache.scalar_type(), "rotary_embedding_cache", [&] {
+          using cache_t = scalar_t;
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:按 diff,所有经由该 CUDA rotary embedding kernel(`pos_encoding_kernels.cu`)的 rope 计算路径:kernel 模板签名变化(编译期接口),cache 不再强制转 fp32;cache 为非 fp32 dtype 时数值路径由「先整块转 fp32」变为「按原 dtype 读取后逐值 static_cast<float>」,精度表现可能变化;bfloat16 参数化测试新增覆盖。

## 关联
- https://github.com/vllm-project/vllm/issues/41017

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
