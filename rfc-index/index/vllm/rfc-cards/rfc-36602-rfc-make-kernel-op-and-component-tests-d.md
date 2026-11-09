# RFC #36602: [RFC]: Make kernel/op and component tests device-agnostic for OOT plugins

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36602](https://github.com/vllm-project/vllm/issues/36602) |
| 作者 | romitjain |
| 创建日期 | 2026-03-10 |
| 状态 | ⚪ closed(关闭于 2026-07-24) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Attention / Kernel |

## 一句话结论
把 vLLM 内核/op 与组件测试去掉 CUDA 硬编码,让 OOT 设备插件直接复用上游测试套件。

## 提案动机
vLLM 的 kernel/op 测试与组件测试对正确性和回归提供了大量覆盖,但许多测试硬编码 CUDA 假设——设备字符串、`opcheck()` 调用、`torch.cuda.*` API——导致 OOT 设备插件无法复用它们,插件开发者被迫为上游已充分测试的 op 和组件重复编写测试逻辑。此前 #20169 已把同样的设备无关化模式应用到 v1 worker 和 sampler 测试(Intel XPU),但 kernel/op 测试一直未处理。

## 提出的方案
整理出测试套件中反复出现的 4 类 CUDA 硬编码模式并逐一替换:
1. `CUDA_DEVICES` 设备列表改由 `current_platform.device_type` 与 `current_platform.device_count()` 生成(涉及 `test_layernorm.py`、`test_activation.py`、`test_pos_encoding.py`、`test_apply_rotary_emb.py` 等);
2. `opcheck()` 改为仅在 `current_platform.is_cuda_alike()` 时调用——它只校验 CUDA custom op 契约,在非 CUDA 设备上会直接失败;
3. 作为参考实现的 `layer.forward_native()` 改用 `layer.forward_static()`——OOT 插件注册自定义 op 后,`forward_native()` 可能派发到覆盖实现而非基类实现,污染参考输出;
4. 组件测试中硬编码的 `torch.device("cuda:0")` 改为 `torch.device(f"{current_platform.device_type}:0")`。

正文给出两个示例:`tests/kernels/core/test_layernorm.py` 同时含模式 1、2、3,已有 PoC PR;`tests/kernels/core/test_activation.py`(覆盖 SiLU、GELU、FatReLU 等激活)含模式 1、2。

## 优化点与收益
正文未给出量化数字。收益为工程性而非性能:OOT 插件只需安装插件即可运行 `pytest` 复用上游同一套测试,对齐上游覆盖,消除重复的测试代码维护成本。

## 现状与落地
已关闭(关闭于 2026-07-24,labels 含 stale)。正文给出的落地材料是 `test_layernorm.py` 的 PoC PR,反馈期设为 2 周;整体落地情况需查关联 PR。

## 关联
- [PoC PR #36246](https://github.com/vllm-project/vllm/pull/36246)(test_layernorm.py)
- [#20169](https://github.com/vllm-project/vllm/pull/20169)(v1 worker/sampler 测试设备无关化的先例)
- [vLLM plugin system 文档](https://docs.vllm.ai/en/latest/design/plugin_system/)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
