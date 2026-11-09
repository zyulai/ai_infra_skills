# PR #25663: [MoE Refactor] [NPU] Refactor Ascend MoE implementation to reduce code duplication and align with community design

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @OrangeRedeng |
| 合并 | 2026-07-15 |
| 分类 | MoE (tier: major) |
| Label | documentation, quant, amd, Multi-modal, deepseek, speculative-decoding, hicache, npu, run-ci, diffusion, model-gateway, jit-kernel |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(47 个文件未收入),收录部分含 5 组件新文件、moe_methods/fuseep/awq_kernels 改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/25663 |

## 一句话结论

把 Ascend NPU MoE 从 10 处重复的 monolithic `apply` 流程重构为 5 组件 + AscendMoERunner/AscendA2A 后端(对齐 #8715 社区设计),顺带带来 W4A4 NZ 格式最高 5–10% 提速等多项修复。

## 动机

Ascend MoE 实现存在两大问题:(1) 代码重复——tokens routing、w13 gate_up 分组 matmul、activation、w2 down_proj 分组 matmul、finalize routing 这条流水线在 unquant/gguf/w4a4/w4a8/w8a8(prefill/decode 分开)/w4a16 等 10 处各自复制一份,任何 kernel 更新或 bug 修复都要同步所有副本;(2) 偏离社区设计——dispatch 与 grouped-GEMM 硬编码在单一 `apply` 流程里,计算与通信耦合,无法接入 #8715 定义的标准 All-to-All backend 与 MoE Runner 模式。

## 优化点

结构上分两层:(1) 5 组件分解——把 MoE 前向拆为 Initial Routing(`init_routing.py`,v1/v2/Quant 三种 npu API)、Hidden States Quant(`hidden_states_quant.py`,动态/静态)、Grouped Matmul(`matmul.py`)、Activation(`activation.py`,NPUSwiglu/SwigluQuant/SwigluQuantWithScales/SwigluDeepEPKernel/GeluAndMul/SwigluOAI/StepAndMul + TP all-gather 包装)、Finalize Routing(`finalize_routing.py`),各自有 `Base*` 抽象接口,runner 按模型配置与量化方案在运行时选型,可对每个 matmul 独立选量化方法;(2) 新建 `AscendMoERunner`(`layers/moe/moe_runner/ascend.py`,matmul1/activation/matmul2 抽成独立可替换 stage,DeepEP 通过 pre/post-permute hook 支持 normal/low-latency 模式)与 `AscendA2ABackend`(`token_dispatcher/ascend_tp.py` 的 AscendTPDispatcher,把 init/finalize routing 从 `apply` 中抽出),`fused_moe_method_npu.py` 的各 `npu_fused_experts_*` 删除或改薄包装,`moe_methods.py` 只留配置与轻量委派(w4a4/w4a8/w8a8/w4a16 方法重写为按 matmul 粒度),decode 专用路径与 prefill 合并(作者称测量未显示分离的优势),`fused_moe_npu()` 标记 deprecated。附带收益:AWQ/GGUF/GPTQ/Auto-round 可与 DeepEP 配合;AWQ Linear 新增绕过 `npu_weight_quant_batchmatmul` 限制的 unquant fallback 路径(group_size 不满足 NPU 约束时去量化为 bf16 F.linear);AWQ MoE 在 `process_weights_after_loading` 重设计后省 2x 内存;ModelSlim 支持混合位宽(w13/w2 不同量化方案);GGUF 经 npu_format_cast 启用 NZ 权重格式;W4A4 使用原生 NZ 内存布局(而非强制 ND)带来最高 **5–10% speedup**;新增 `SGLANG_NPU_W4A4_NEW_PACKING=True` 支持 int4x2 打包(两个 int4 存一个 int8),权重加载内存减半;并 revert #29503 以避免性能回退。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/moe/{init_routing,hidden_states_quant,matmul,activation,finalize_routing}.py: 新增,MoE 前向 5 组件抽象
python/sglang/srt/layers/moe/moe_runner/ascend.py: 新增,AscendMoERunner(stage 化执行 + DeepEP pre/post-permute hooks)
python/sglang/srt/layers/moe/token_dispatcher/ascend_tp.py: 新增,AscendTPDispatcher(NPU A2A 后端)
python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py: 新增,各量化方案的轻量 MoE method 配置
python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py: AWQ Linear/MoE 重写(NPU 快路径约束检查 + bf16 fallback)
```

```python
class GroupedMatmul(BaseMatmul):
    def forward(self, layer, weight_prefix, hidden_states, expert_tokens, ...):
        weight = getattr(layer, f"{weight_prefix}_weight", None)
        return torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states], weight=[weight] if transposed else [weight.transpose(1, 2)],
            split_item=2, group_list_type=group_list_type, group_type=0,
            group_list=expert_tokens, output_dtype=output_dtype, ...)[0]
```

## 性能声明

- **W4A4 NZ 格式修复:最高 5–10% speedup**(NPU kernel 改用原生 NZ 内存布局,不再强制 ND)。
- 其他性能类声明(无独立数字):AWQ MoE `process_weights_after_loading` 后内存占用减半;int4x2 打包使权重加载内存减半;GGUF(NZ 格式)与 GPTQ(新 kernel 版本)性能改进。
- Accuracy Tests 与 Speed Tests 两节均为图片(截图),文本数字不可见;perf claim 正则还检测到一处 "10% speedup" 字样来自 W4A4 相关描述区间。PR 未注明测试硬件与模型细节(仅可推断为 Ascend NPU)。

## 验证与回归风险

5 组件与 runner 使路由/激活/量化逻辑可独立单测(作者强调 testability);`fuseep_mode` 从环境变量改为 server_args;`fused_moe_npu()` 保留 deprecation warning 以兼容未迁移 FusedMoE 的模型。风险面:该重构触及 NPU 全部 MoE 量化路径(Unquant/AWQ/GPTQ/ModelSlim/CompressedTensors/GGUF)与所有 dispatcher(NPU native、DeepEP normal/low-latency、FuseEP),行为等价性依赖新组件装配;decode/prefill 路径合并后 decode 专用优化消失(作者称测量无优势);AWQ fallback 路径有额外内存开销(去量化权重)。CI 状态:Base 运行中(hourglass),Extra 失败(:x:)。

## 关联

设计依据:issue #8715(社区 MoE Runner/All-to-All 模式);revert:#29503;同区域:NPU MoE 与 DeepEP/FuseEP 相关 PR;pack 未列出其他具体关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
