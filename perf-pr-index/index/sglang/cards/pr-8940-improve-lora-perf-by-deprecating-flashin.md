# PR #8940: Improve LoRA Perf by Deprecating FlashInfer and Eliminating Redundant Tensor Ops

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @lifuhuang |
| 合并 | 2025-08-10 |
| 分类 | Attention / Kernel | tier: major |
| Label | performance, lora, ready-to-merge |
| Diff | diffstat 缺失(素材未提供统计数字);diff 含约 10 个文件(另注明 3 个文件因超长被裁剪),整体为删代码为主的重构,删除 flashinfer_backend.py 全文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/8940 |

## 一句话结论

弃用 FlashInfer LoRA backend 并重构 LoRA B 权重布局与批次准备,ITL P95 降低 29.8%。

## 动机

此前 LoRA 路径存在两块开销:一是 FlashInfer LoRA backend(#7809)功能受限(单 rank/单 scaling)且迫使其他代码路径为兼容它保留大量冗余显存操作(#6961、#8712);二是 `prepare_lora_batch` 每个 micro-batch 都要执行 `set_lora_info`,其中包含大量 reshape/copy 等 redundant mem ops。PR 目标是移除 FlashInfer backend、统一到 triton kernel 期望的张量布局,并把每 batch 的重复工作移到服务初始化阶段。

## 优化点

三方面机制改动:(1) 直接删除 `flashinfer_backend.py`,选择 flashinfer 后端时抛错提示改用 triton;同时删掉 `fuse_output_add`/`fuse_stacked_lora_b` 这两个仅为区分两套后端而存在的开关分支。(2) LoRA B 权重改为按 triton kernel 期望的 stacked 布局存储(q+k+v 沿输出维 concat,shape 去掉 stacked_multiply 维),`normalize_qkv_proj`/`normalize_gate_up_proj` 不再为 kv_proj 拆分再 stack,`slice_lora_b_weights` 在加载期一次完成 TP 分片并 concat,从而 `set_lora_info` 里所有运行期 copy/stack 操作全部消失,`apply_lora` 把 `base_output` 直接传给 kernel 由 triton kernel 内融合 add。(3) `update_lora_info()` 从每个 batch 的 `transfer_adapter_info` 移到 `init_state`(服务初始化)阶段调用,避免每 batch 重复绑定 memory buffer。附带修复 triton `qkv_lora_b` kernel 中 mask 用 `and` 代替 `&` 的 bug,并简化 `get_normalized_lora_weight_names` 为单一集合映射。

## 关键改动

```text
python/sglang/srt/lora/backend/flashinfer_backend.py: 整文件删除,弃用 FlashInfer LoRA 后端
python/sglang/srt/lora/backend/base_backend.py: 删除 fuse_output_add/fuse_stacked_lora_b 分支,flashinfer 选择报错
python/sglang/srt/lora/layers.py: set_lora_info 去掉 stack/copy,apply_lora 传 base_output 给 kernel 融合 add
python/sglang/srt/lora/lora_manager.py: update_lora_info 从每 batch 移到 init_state 一次执行
python/sglang/srt/lora/mem_pool.py: B buffer shape 去掉 stacked 维,加载/视图逻辑简化
python/sglang/srt/lora/utils.py: 权重名归一化改为单集合,get_hidden_dim 合并 qkv/gate_up 维度
```

关键片段(简化后的 apply_lora):

```python
def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
    lora_output = self.lora_backend.run_lora_b_sgemm(
        x=lora_a_output,
        weights=self.B_buffer,
        base_output=base_output,
    )
    return lora_output
```

## 性能声明

benchmark 条件:Llama-3.1-8B-Instruct,`--disable-radix-cache --lora-paths lora=algoprog/fact-generation-llama-3.1-8b-instruct-lora`,`bench_serving --num-prompt 480 --request-rate 8 --lora-name lora`(PR 未注明 GPU 型号与精度)。结果:

- ITL P95:54.69ms → 38.40ms(-29.8%)
- ITL P50:23.35ms → 18.30ms(-21.6%)
- E2E Latency P50:2996.55ms → 2323.63ms(-22%)

Profile:`prepare_lora_batch` 耗时 3.162ms → 162us(约 19.5 倍)。准确性由 CI 覆盖,PR 未附单独数字。

## 验证与回归风险

准确性声明"Covered by CI",素材 diff 中未见新增专项测试文件(3 个文件被裁剪,可能含测试改动)。回归风险:flashinfer LoRA 后端用户升级后会直接报错,需迁移到 triton 后端;LoRA B 权重布局与 memory pool shape 变化触及加载、TP 分片、buffer 视图多条路径,q/kv 异构 rank 场景此前要求 q/kv rank 一致才能 stack,重构后依赖加载期 concat,极端配置(如仅 k_proj 无 lora 时 zero-fill)需 CI 覆盖。

## 关联

前序/动机:#7809(deprecate FlashInfer LoRA backend)、#6961、#8712(prepare_lora_batch 冗余操作);提及 #8939(bug,标注 rebase 依赖 #8987)。同区域:LoRA 后续工作如 #14105(MoE 层 LoRA 支持)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
