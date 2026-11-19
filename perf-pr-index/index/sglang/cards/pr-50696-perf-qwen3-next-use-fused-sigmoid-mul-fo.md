# PR #50696: [Perf] qwen3-next: Use fused_sigmoid_mul for the attention output gate

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jzlxiaohei |
| 合并 | 2026-11-16 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/50696 |

## 一句话结论

Qwen3-Next 的 attention 输出门控在 CUDA 上改用已有的 `fused_sigmoid_mul` 直接吃 `[T, H, D]` strided gate 视图,免掉 gate 拷贝并合并 sigmoid+乘法两个内核:微基准 2.0×–3.9× 提速且输出逐位一致(bit-identical),TP1 decode 每步约省 80 us。

## 动机

`Qwen3HybridAttentionDecoderLayer` 中,attention 输出门控在 CUDA 上先用 `gate.reshape(...)` 拷贝 strided gate 切片,再跑 `torch.sigmoid` 与一次乘法共三个内核;而 `fused_sigmoid_mul` 本就支持直接读 3D strided gate——Qwen3.5(`qwen3_5.py`)同层形状已在用它,HIP 分支此前也是在拷贝之后用它。

## 优化点

- `qwen3_next.py` 的 `forward_prepare_native` 保留 `torch.chunk` 产出的 `[T, H, D]` strided view,不再 reshape 拷贝。
- CUDA:`fused_sigmoid_mul(attn_output, gate, inplace=True, round_sigmoid=True)`;`round_sigmoid` 在乘法前把 sigmoid 结果舍入到 bf16,使输出与原 `attn_output * torch.sigmoid(gate)` 逐位一致。
- HIP:同内核同参数(无 `round_sigmoid`),仅去掉 gate 拷贝,输出不变。
- CPU / NPU:保持 torch 路径(3D gate 先 reshape 回 2D)。
- 顺带删除重复的 `_use_aiter` 定义;新增 head-strided gate 的 bit-exact 测试;`test_qwen3_next_fused_qk_prologue.py` 改用 `reshape_as` 比较现在为 3D 的 gate。

## 关键改动

```python
# qwen3_next.py
-            gate = gate.reshape(*orig_shape, -1)
+            # gate stays as 3D strided view; fused_sigmoid_mul handles it directly
...
-            if _is_hip:
-                from sglang.kernels.ops.elementwise.elementwise import (
-                    fused_sigmoid_mul,
-                )
-                attn_output = fused_sigmoid_mul(attn_output, gate)
+            if _is_cuda or _is_hip:
+                # round_sigmoid keeps CUDA bit-identical to torch sigmoid-then-mul.
+                attn_output = fused_sigmoid_mul(
+                    attn_output, gate, inplace=True, round_sigmoid=_is_cuda
                 )
             else:
-                gate = torch.sigmoid(gate)
-                attn_output = attn_output * gate
+                gate = gate.reshape(gate.shape[0], -1)
+                attn_output = attn_output * torch.sigmoid(gate)
```

## 性能声明

(全部照抄 PR 正文,未独立核实)

- 微基准:1×A100-SXM4-80GB(driver 595.91.07),bf16,torch 2.14.1+cu130,triton 3.8.0;Qwen3-Next-80B-A3B 单个全注意力层的输出门算子(16 q/2 kv heads, head_dim 256),TP1/TP2/TP4 每 rank 形状,T=1 到 8192,`marker.do_bench`(CUDA graph、冷 L2、取中位,单位 us)。before→after 摘录:TP1 16/2 头 T=1 8.89→2.27、T=128 12.56→4.40、T=8192 341.2→118.2;TP2 8/1 T=1 6.91→1.78、T=8192 176.9→60.8;TP4 4/1 T=1 6.60→1.72、T=8192 85.7→32.8。每种形状 2.0x–3.9x 更快,所有输出 bit-exact。
- 换算(作者陈述):Qwen3-Next-80B-A3B 有 12 个全注意力层(共 48 层,`full_attention_interval` 4);TP1 decode 约每步省 12 × 6.6 us ≈ 80 us,8192-token prefill 约省 2.7 ms。
- 作者未跑端到端 serving 基准:bf16 80B 模型超出其单卡内存。

## 验证与回归风险

正确性靠 bit-exact 而非 eval:`test_sigmoid_gate_mul.py` 24 个测试在 1×A100 全过,新增 head-strided 场景用 `torch.equal` 断言 Qwen3-Next 布局(per-head `[q | gate]`,16 heads × 256);基准脚本对每个形状都检查 `torch.equal(after, before)` 全为 True;prologue 测试改为 `reshape_as` 比较。风险面:改动模型前向热点路径但输出逐位一致;CPU/NPU 走原 torch 路径;HIP 参数不变。CI Base 通过,Extra 与 AMD 显示失败图标(pack 未提供细节)。

## 关联

`fused_sigmoid_mul`(`sglang.kernels.ops.elementwise.elementwise`,已存在内核)、`qwen3_5.py`(同层形状的先例用法)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
