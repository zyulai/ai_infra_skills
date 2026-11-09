# PR #13348: Fix dpsk-r1-fp4 tp8 by reverting two commits (#13162 and #13341)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13348](https://github.com/sgl-project/sglang/pull/13348) |
| 作者 | Qiaolin-Yu |
| 合并日期 | 2025-11-16 |
| labels | high priority, quant, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
DeepSeek-R1-0528-FP4 tp8 + EAGLE speculative V2 启动 hang/fail,回退 #13162、#13341 两个提交修复。

## 问题现象
正文 Motivation 写明:在本 PR 之前,以下命令会 hang/fail——
```
SGLANG_ENABLE_SPEC_V2=1 /root/.python/sglang/bin/python -m sglang.launch_server --model-path nvidia/DeepSeek-R1-0528-FP4-v2 --trust-remote-code --quantization modelopt_fp4 --tp 8  --speculative-algorithm=EAGLE  --port 40020   --kv-cache-dtype fp8_e4m3   --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```
即 DeepSeek-R1 FP4 模型在 tp8、modelopt_fp4 量化、EAGLE 投机(SPEC_V2)与 fp8 KV cache 组合下无法正常启动。具体报错日志正文未附。

## 根因
正文未披露根因,修复方式是整体回退 #13162 与 #13341 两个提交(见标题)。(由 diff 推断)被回退的改动涉及:① `eplb/expert_location.py` 中 `logical_to_all_physical` 新增的 `require_global_experts` 参数及"返回全部物理专家"分支,以及 nearest expert 替换映射的逻辑调整;② 多处 FP4/MoE kernel 调用相关的 `tile_tokens_dim` 参数处理。回退后 `logical_to_all_physical` 恢复为仅按 layer_id/logical_expert_id 返回物理映射,weight_loader 不再读取 `_sglang_require_global_experts` 属性,kernel 调用显式传 `tile_tokens_dim=None`。

## 修复方案
按标题回退 #13162、#13341。diff 实际改动:
- `eplb/expert_location.py`:`logical_to_all_physical` 移除 `require_global_experts` 形参及对应全量物理专家返回分支;`_compute_logical_to_all_physical_map` 中 nearest expert 替换恢复为 `nearest_expert not in mapped_physical_experts` 时写 `mapped_physical_experts[0]`。
- `layers/moe/fused_moe_triton/layer.py`:weight_loader 删除 `require_global_experts = getattr(param, "_sglang_require_global_experts", False)` 相关逻辑;`forward` 调用增加 `tile_tokens_dim=None`。
- `layers/quantization/fp8.py`、`modelopt_quant.py`、`mxfp4.py`:apply 调用均补 `tile_tokens_dim=None`。

关键 diff 摘录(expert_location.py):
```diff
-        self,
-        layer_id: int,
-        logical_expert_id: int,
-        require_global_experts: bool = False,
+        self, layer_id: int, logical_expert_id: int
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本区间,即 #13162/#13341 合入后至本 PR 之间。
- 回归面:改动覆盖 eplb 专家位置映射(EPLB/专家重分配相关路径)、fused MoE triton 层 weight_loader,以及 fp8 / modelopt_fp4 / mxfp4 量化的 MoE apply 调用。使用 require_global_experts 语义的调用方与依赖 `tile_tokens_dim` 新行为的路径会回到旧逻辑;FP4 量化 + 投机解码的 DeepSeek 部署路径是主要受益面。

## 关联
- [#13162](https://github.com/sgl-project/sglang/pull/13162)(标题提到的被回退提交)
- [#13341](https://github.com/sgl-project/sglang/pull/13341)(标题提到的被回退提交)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
