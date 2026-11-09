# PR #61749: [Bugfix][MoE] Require exact strides for moe_forward custom ops

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61749](https://github.com/vllm-project/vllm/pull/61749) |
| 作者 | ldkhang1201 |
| 合并日期 | 2026-10-22 |
| labels | bug, ready, nvidia |
| 热度 | 👍 0 · 💬 46 |
| 分类 | MoE |
| bug 类型 | general |

## 一句话结论
torch.compile 下 hidden 2688 被 pad 到 2816 后,`needs_fixed_stride_order` 允许 Inductor 把 unpadded 激活作为跨行 view 传给 NVFP4 quant kernel,逐行读错输出乱码;改 tag 为 `needs_exact_strides`。

## 问题现象
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` 在 torch.compile + FlashInfer TRTLLM NVFP4 MoE 后端下输出乱码(garbage),`--enforce-eager` 正常。GB300 实测 GSM8K(1319 题):eager completions 0.303 / chat 0.882;compile 下 main 为 0.006(46% invalid)/ 0.008(47% invalid),本 PR 恢复到 0.299 / 0.890(eager parity)。

## 根因
正文披露:Nemotron-3-Nano hidden size 为 2688,TRTLLM NVFP4 MoE 将其 pad 到 2816,`MoERunner._maybe_pad_hidden_states` 在编译图内施加 `F.pad`;Inductor 把前级 RMSNorm 融合进 padded buffer。由于 `needs_fixed_stride_order` 只固定 stride **顺序**而非布局,Inductor 把 `shared_experts_input`(未 pad 的 2688 宽激活)作为该 buffer 的 view 传入 `moe_forward_shared`:`reinterpret_tensor(buf, (s, 2688), (2816, 1), 0)`。共享专家的 NVFP4 linear 调 `scaled_fp4_quant`,其 kernel 假定行是 packed 的——第 0 行读对,之后每一行都读错偏移。eager 模式从不产生该 alias。

## 修复方案
两行改动(`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`):`vllm.moe_forward` 与 `vllm.moe_forward_shared` 两个 custom op 的 tag 从 `torch.Tag.needs_fixed_stride_order` 改为 `torch.Tag.needs_exact_strides`,生成代码改为传入独立的 packed buffer。选该 tag 的理由:它是 torch 2.6 时代的 workaround,#19346/#28974/#33209 已在其他 MoE custom op 上移除,这两个漏掉了;torch 2.8 起 `needs_exact_strides` 本就是 custom op 默认(pytorch/pytorch#148063、#150511),此处只是恢复默认。关键 diff 摘录:
```python
-    tags=(torch.Tag.needs_fixed_stride_order,),
+    tags=(torch.Tag.needs_exact_strides,),
```
新增单测 `test_moe_forward_shared_gets_packed_shared_input_under_inductor`(模拟 hidden 2688/pad 2818 无关、2688→2816 场景):旧 tag 下失败于 `[(2816, 1)] == [(2688, 1)]`,加修复后 5 passed。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响 torch.compile 下走这两个 op 的路径(hidden dim 被 pad 的场景尤甚)。
- 回归面:仅这两行 tag;无其他代码路径。代价:Inductor 首选布局与 eager 不同时可能插入一次拷贝——实践上是 hidden pad 时每 MoE 层一次 `(tokens, hidden)` copy;不 pad 时输入已 contiguous,无变化;不引入 host sync。ROCm gpt-oss 的 AITER RMSNorm+pad fusion 传给 `moe_forward` 的是 padded buffer 本身(contiguous),不受影响。**缓存注意**:改 tag 不会使 Inductor FX graph cache 或 vLLM AOT compile cache 失效,`VLLM_DISABLE_COMPILE_CACHE=1` 也不旁路 Inductor cache;同版本 dev install 可能继续加载旧 alias 图,需清缓存(版本升级不受影响)。遗留:`needs_fixed_stride_order` 仍在 `transformers/moe.py`、`models/utils.py`、`qwen3_dflash2.py`,本 PR 未分析、未改动。

## 关联
- #52308(所修 issue)
- #52383(作者自己关闭的前一方案,基于错误诊断,会破坏 #34636)
- #34636(ROCm RMSNorm+pad fusion,前方案会破坏它)
- #49092(`scaled_fp4_quant` 加 `.contiguous()`,kernel 侧互补防护)
- #19346、#28974、#33209(其他 MoE op 移除该 tag 的先例);#61179、#61367(同 op 注册、不同 bug 的开放 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
