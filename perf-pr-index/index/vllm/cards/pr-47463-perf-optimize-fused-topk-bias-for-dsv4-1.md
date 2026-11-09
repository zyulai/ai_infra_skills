# PR #47463: [Perf] Optimize `fused_topk_bias` for DSv4, 1.5~2x kernel performance improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-07-15 |
| 分类 | MoE(major) |
| Label | ready |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 CUDA 内核、router 与测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/47463 |

## 一句话结论

把 input_ids/hash 表的 dtype 转换移进 fused_topk_bias 的 CUDA 内核,DSv4 hash MoE 路由内核提速 1.61–2.01x。

## 动机

原实现在 hash MoE 路径下要求 input_ids/tid2eid 与 topk_ids 同 dtype,Python 侧每次 forward 都做冗余的 input_tokens.to(indices_type) 与 hash 表 to(int32) 转换。属于 issue #45861 的一部分。

## 优化点

CUDA 内核 topkGatingSoftplusSqrt 增加独立模板参数 HashIndType:input_ids 与 tid2eid 可保持 int64(或 int),由新增的 load_index_as_int64 设备函数在内核内读取并转换,输出 indices 仍按 IndType 写入;hash 查表索引 token_id*k 也改为 int64 计算。Python 侧 fused_topk_bias 只在 input_tokens 与 hash_indices_table dtype 不一致时才转换 input_tokens,不再强制转成 indices_type。launcher 用 lambda 按 topk_indices 的 int/uint32/int64 分派,并校验 input_ids 与 tid2eid dtype 一致。

## 关键改动

```text
csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu: HashIndType 模板参数 +
  load_index_as_int64 内核内转换,hash 路径按 int64 寻址
vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:
  移除 Python 侧对 input_tokens/hash_indices_table 的强制 dtype 转换
tests/kernels/moe/test_topk_softplus_sqrt.py: hash 测试改用 long 表与 long input_ids
```

## 性能声明

microbench(vocab=163840,experts=512,topk=8,int32 indices,AI 生成脚本):old_python_cast vs current_mixed(ms/forward):M=128 0.0311→0.0163(1.91x);M=512 0.0316→0.0158(2.01x);M=2048 0.0308→0.0162(1.91x);M=8192 0.0313→0.0195(1.61x);与 precast int32 基本持平。标题口径为 1.5~2x。测试硬件 PR 未注明(脚本要求 CUDA)。

## 验证与回归风险

精度由现有单元测试覆盖(test 改用 long 输入,断言新旧 ids 精确一致、weights 近似一致)。风险限于 DSv4 hash MoE 路由内核:新增模板实例化扩大编译产物;非 hash 路径传 nullptr 指针,行为不变。

## 关联

issue #45861 的组成部分;pack 未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
