# PR #27244: [CPU]Improve cpu fused moe perf

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xiangze-arm |
| 合并 | 2025-11-06 |
| 分类 | MoE (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 1 个文件(cpu_fused_moe.py) |
| 链接 | https://github.com/vllm-project/vllm/pull/27244 |

## 一句话结论

为 AArch64 CPU fused MoE 接入 oneDNN/ACL matmul 路径(逐专家预建句柄),32 核 Neoverse-N2 上吞吐约 1.6x。

## 动机

CPU fused MoE(`CPUFusedMOE`)此前对每个专家的 gate_up 与 down 两次投影直接调 `F.linear`,即裸 torch eager matmul,未利用 oneDNN/ACL 在 AArch64 上的优化 GEMM 路径;且每次调用都要按专家索引取权重、判断 bias 是否存在,属于每步重复的准备开销。

## 优化点

把逐专家的 GEMM 调用从"每次前向时取权重 + F.linear"改为初始化时一次性预建的可调用列表。构造 `CPUFusedMOE` 时探测 `ops._supports_onednn and ops.is_onednn_acl_supported()`:支持则对每个专家用 `ops.create_onednn_mm(layer_w13_weight.t(), 32)` / `create_onednn_mm(layer_w2_weight.t(), 32)` 创建 matmul 句柄,封装成 `lambda x, handle=..., bias=...: ops.onednn_mm(handle, x, bias)` 存入 `layer.gate_up_linear` / `layer.down_linear`;不支持则回退为 `F.linear` 闭包,行为与原来一致。onednn 路径下权重被提前转置进句柄,原 `w13_weight`/`w2_weight` 被替换为空 Parameter 释放引用。前向循环只剩 `layer.gate_up_linear[i](tokens)`、激活、`layer.down_linear[i](gate_up)`,消除每步的权重索引与属性查找。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/cpu_fused_moe.py: CPUFusedMOE.__init__ 预建逐专家 onednn_mm/F.linear 闭包;__call__ 改用预建列表
```

句柄预建(节选):

```python
if use_onednn_mm:
    gate_up_handle = ops.create_onednn_mm(layer_w13_weight.t(), 32)
    layer.gate_up_linear.append(
        lambda x, handle=gate_up_handle, bias=layer_w13_bias: ops.onednn_mm(
            handle, x, bias
        )
    )
    ...
    layer.w13_weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
```

## 性能声明

PR 正文声明:MoE 在 AArch64 CPU 上走 onednn_mm 路径后,32 个 Neoverse-N2 核上吞吐约 1.6x(相对当前默认路径)。测试条件:`Qwen3-30B-A3B`,bench 命令 `vllm bench throughput --dtype=bfloat16 --num-prompts 64 --seed 0 --dataset-name sharegpt --max-model-len=2048 --input_len=64`;正文未给出绝对 tok/s 数字,也未注明 CPU 型号细节(仅"32 Neoverse-N2 cores")。

## 验证与回归风险

验证:本地 Qwen3-30B-A3B e2e 测试正常;单元测试 `pytest tests/kernels/moe/test_moe.py -k test_cpu_fused_moe_basic` 通过。回归风险:仅影响 CPU fused MoE 路径;onednn 分支要求 `_supports_onednn` 且 ACL 后端可用,不满足时走 F.linear 原路径(无行为变化);权重转置发生在 init(`w13_weight.t()`),权重加载/量化流程若在 init 之后修改权重会与句柄脱钩——onednn 路径下原权重 Parameter 被置空,依赖按名访问 `w13_weight` 的下游(如量化、权重加载、检查点导出)需要注意;`create_onednn_mm` 的第二个参数 32(batch size 相关配置)为硬编码。

## 关联

同区域 CPU MoE 优化:PR #48637(INT8 Fused MoE Kernel for Arm CPUs,同作者的后续方向)、PR #37469(CPU BF16 GELU LUT)。PR 正文未引用其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
