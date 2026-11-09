# RFC #48941: [RFC] Recursive Tensor Collector: Fix CUDA Graph Invalidation After Weight Reload

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48941](https://github.com/vllm-project/vllm/issues/48941) |
| 作者 | new-TonyWang |
| 创建日期 | 2026-07-17 |
| 状态 | ⚪ closed(关闭于 2026-07-21) |
| labels | rocm |
| 热度 | 👍 0 · 💬 3 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议递归张量收集器:权重重载后把派生张量新值拷回旧地址,系统性修复 CUDA graph 因地址漂移而失效的一整类 bug。

## 提案动机
`process_weights_after_loading()` 创建的派生张量(workspace、stride 描述符、fused scales、排列索引)以普通 Python 属性存放在 `nn.Module`、kernel 对象或 quant method 实例上,不在 `_parameters`/`_buffers` 内,因此对现有 copy-back 机制不可见。reload 后这些张量获得新地址,而 CUDA graph 在捕获时已烧录旧 `data_ptr()`,replay 只按记录地址读写,于是读到过期/已释放内存:轻则静默错误输出,重则 illegal memory access 崩溃,或 Marlin workspace(原子计数器)读到垃圾值导致内核无限等待——正文确认 RTX 4090 上 100% GPU 占用超 10 分钟的 livelock。静态审计 + 实测确认 13 处受影响点(MLA W_UV/W_UK_T、CUTLASS FP8 MoE strides、Marlin workspace 及 8 个 Marlin 兄弟变体、AITER/FlashInfer/compressed-tensors 等),逐个修补是 whack-a-mole,需要基础设施级系统性方案。

## 提出的方案
- `tensor_collector.py` 的 `collect_extra_tensors(layer)`:递归遍历层的属性树,收集所有不在 `_parameters`/`_buffers` 内的 CUDA 张量,覆盖普通张量属性、嵌套 Python 对象(kernel、quant_method、experts)、dict/list/tuple 内、`functools.partial` 参数、函数闭包;不递归进子 `nn.Module`;按 storage id 去重,最大深度 8(实际路径约 4-5 层)。
- 每个张量记录为 `TensorSlot`(path、tensor 强引用防 GC、data_ptr、storage_id);`path_resolver.py` 按 dotted path(含 dict/list 索引)导航并回写对象图。
- 三处外科手术式集成进 `layerwise.py`:initialize 阶段收集 extra tensor slots;copy-back 阶段 `_copy_back_extra_tensors()` 对 shape+dtype 兼容的 slot 执行 `slot.tensor.data.copy_(new_tensor)`(新值写进旧地址)并用 `set_by_path()` 把旧张量接回对象图。
- 整个对象被替换时依然有效:旧张量由强引用保住,`resolve_path` 定位到新对象属性,新值写入旧地址,新对象属性改指旧张量,CUDA graph replay 地址不变。
- 属性:零改动 kernel/quant/attention 代码;新 kernel 自动被覆盖(前向兼容);shape/dtype 不匹配则记日志跳过并发出重捕获信号;额外强引用仅针对小张量,内存开销相对模型权重可忽略;共享 storage 别名(如 `c_strides2 = ab_strides1`)去重处理。

## 优化点与收益
正文未给出吞吐类量化数字。给出的收益与验证:消除 13 处确认/高风险点的整类 bug;H200 端到端真实模型 reload + 生成测试 20 项全 PASS——DeepSeek-V3-debug FP8 MoE strides(19 extras tracked,0 drifted)、MLA W_UV/W_UK_T(6)、Machete W4A16(114)、Qwen3-0.6B BF16 基线(30)、FP8(114)均 0 drifted;运行时数值刷新 4 项、MoE kernel 连续 2 次 reload 2 项、上游 `test_reload.py` CPU 测试 9/9 PASS;mock 层单元测试 13 项初步 PASS。

## 现状与落地
已关闭(关闭于 2026-07-21,创建后约 4 天)。正文含 Validation Status 章节,H200 端到端测试与单元测试均已跑通,但正文未写明实现 PR 是否合并;落地情况需查关联 PR(#48251、#48438、#48539 等)。注:pack 正文在 20k 字符处截断(「Ongoing: Unit Tests with Mock Layers」段落中途截断),本卡基于现有内容生成。

## 关联
- Related Issue: https://github.com/vllm-project/vllm/issues/48312
- PR: https://github.com/vllm-project/vllm/pull/48251
- PR: https://github.com/vllm-project/vllm/pull/48438
- PR: https://github.com/vllm-project/vllm/pull/48539
- Issue: https://github.com/vllm-project/vllm/issues/41670

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
