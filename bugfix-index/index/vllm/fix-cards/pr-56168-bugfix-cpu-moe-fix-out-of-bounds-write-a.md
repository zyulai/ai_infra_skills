# PR #56168: [Bugfix][CPU][MoE] Fix out-of-bounds write and segfault when router weights are fp32

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56168](https://github.com/vllm-project/vllm/pull/56168) |
| 作者 | farzad-elastix |
| 合并日期 | 2026-09-24 |
| labels | bug, verified |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
CPU 后端 fp32 路由权重的 MoE 模型,权重移除时占位张量丢失 dtype,bf16 激活直接喂 fp32 GEMM 造成堆越界/SIGSEGV;三处占位补显式 dtype。

## 问题现象
CPU 后端上,router 权重为 fp32 的 MoE 模型在每次 router GEMM 时破坏堆(heap corruption),进程以 SIGSEGV 或 glibc abort 死亡。Hunyuan-V3 在 warmup 阶段就崩溃,一个请求都没回答;MiniMax-M2 与 Nemotron-H 以同样方式构建 router,同样受影响。只设 `out_dtype=torch.float32` 的模型权重仍是 bf16,不受影响。手动复现:在 `set_default_torch_dtype(torch.bfloat16)` 下对 fp32 `torch.nn.Linear(4096, 192)` 执行 `dispatch_cpu_unquantized_gemm(remove_weight=True)` 后 `layer.weight.dtype` 变成 bfloat16,再以 bf16 输入跑 `cpu_unquantized_gemm` 会向 384 字节缓冲区写 768 字节。

## 根因
正文披露三步链条:
1. `GateLinear` 以 fp32 存 router 权重,通过比较 `x.dtype` 与 `self.weight.dtype` 决定是否上转 bf16 激活。
2. `dispatch_cpu_unquantized_gemm` 用真实 fp32 权重构建 GEMM primitive 后,`remove_weight=True` 时把 `layer.weight` 重绑到不带 `dtype` 的占位 `torch.empty(0)`——原权重 dtype 被遗忘,替换为环境默认 dtype(如 base_loader.py 加载模型时设置的 bf16)。
3. 前向遇 bf16 输入时 `GateLinear` 跳过上转,把 bf16 `x` 传给 `ops.onednn_mm`;该调用按 `x.dtype` 分配输出缓冲,而被调 primitive 仍按 fp32 写入,产生非法内存访问。
关联信息:#27465 修过通向同一破坏的另一条路径,并点破了此处被打破的假设——"同一个推理实例(进程)内所有 handler 应使用同一 dtype"。

## 修复方案
改 `vllm/model_executor/layers/utils.py` 的 `dispatch_cpu_unquantized_gemm`:在三处 `remove_weight` 分支(zentorch_linear_unary、sgl-kernel weight_packed_linear、oneDNN onednn_mm)把占位参数改为 `torch.empty(0, dtype=dtype, device=layer.weight.device)`,复用各分支上方已捕获的 `dtype`,使占位张量保持原权重 dtype,第 1 步的类型比较得以对真实 dtype 生效。新增参数化测试 `test_dispatch_cpu_unquantized_gemm_remove_weight_keeps_dtype`,对 bf16/fp16/fp32 三种权重 dtype 经 `_mock_zentorch_linear_unary` fixture 走 zentorch 分支,断言占位权重 dtype 不变且前向无 NaN,任何主机可确定性运行。

```diff
         if remove_weight:
-            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
+            layer.weight = torch.nn.Parameter(
+                torch.empty(0, dtype=dtype, device=layer.weight.device),
+                requires_grad=False,
+            )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`vllm/model_executor/layers/utils.py` 的 CPU 未量化 GEMM dispatch 三条后端分支(zentorch/sgl-kernel/onednn)的权重移除行为;此前占位 dtype 恒为环境默认,现保留原 dtype,对本来就是默认 dtype 的权重无差异;`remove_weight=False` 与 GPU 路径未动。验证:新测试修复前 3 failed/3 passed、修复后 6 passed;手动脚本不再 abort;4 层 Hunyuan-V3 serve 从 warmup 崩溃变为可完成请求。

## 关联
- #27465(同一破坏的另一条路径的修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
