# PR #12078: [Ascend] qwen optimization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Liwansi |
| 合并 | 2025-11-25 |
| 分类 | KV Cache / Memory(major) |
| Label | npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数;diff 另有 2 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/12078 |

## 一句话结论

Ascend 上 Qwen 系列的一揽子优化:新增 ascend_fuseep MoE A2A 后端(decode 集成 dispatch/gmm1/swiglu/gmm2/combine),Qwen3 235B 性能提升 10%。

## 动机

关联 #10337。Ascend 后端此前存在若干资源与调度 bug(w8a8 int8 权重双份显存、CMO 流同步在 BS=1/2 时可能死锁、eplb index 算子缺失),且 Qwen3 的 qk-norm+RoPE、topk 归一化、PA-图的路径均未用上 NPU 的高性能算子,MoE all-to-all 缺少融合后端。

## 优化点

bugfix 三项:`w8a8_int8.py` 的 w13/w2 权重 transpose+contiguous 不再双份占显存(`release_weight_cache` 用 `resize_(0)` 释放旧 storage);`common.py` 的 CMO 流改为显式 set/wait,修复 BS=1、2 时 stream sync 等待造成的死锁;eplb 在未设 map path 时也引入 index 算子。优化四项:Qwen3/Qwen3MoE attention 改用 triton 融合算子 `split_qkv_rmsnorm_rope`(第 0 层预取 cos/sin,跨层复用 `position_cos/position_sin`);`topk.py` 用 `l1_norm` triton 算子替代 sum+除法归一(注意改为先归一再 cast fp32);`npu_graph_runner.py`/`ascend_backend.py` 让 PA(paged attention)进入 NPUGraph,按 `bs < attention_tp_size` 选择 fused infer attention 或 `_npu_paged_attention` 路径;新增 `ascend_fuseep` moe-a2a-backend(`NpuFuseEPMoE` + `NpuFuseEPDispatcher`),仅支持 PD 分离下的 decode,把 dispatch、gmm1、swiglu、gmm2、combine 融合为单个 `fused_deep_moe` 调用,权重加载后做 format cast(29/NZ)与 scale 重排。

## 关键改动

```text
python/sglang/srt/layers/moe/token_dispatcher/fuseep.py: 新增 NpuFuseEPDispatcher(fused_deep_moe 单调用)
python/sglang/srt/layers/moe/ep_moe/layer.py: 新增 NpuFuseEPMoE(权重 NZ format cast、w13 reshape、scale permute)
python/sglang/srt/models/qwen3.py, qwen3_moe.py: NPU 走 split_qkv_rmsnorm_rope 融合路径
python/sglang/srt/layers/attention/ascend_backend.py, model_executor/npu_graph_runner.py: PA 接入 NPUGraph
python/sglang/srt/layers/quantization/w8a8_int8.py: 权重缓存显存修复;ENABLE_ASCEND_MOE_NZ 下 w 权重 NZ 格式
```

## 性能声明

文本声明:`ascend_fuseep` 后端在 Qwen3 235B 上"Performance improved by 10%"。PR 的 Accuracy 与 Benchmarking 证据为四张截图(Qwen-32B 精度、Qwen-235B 精度、Qwen-235B benchmark、fuseep vs deepep 对比),截图内数字在素材中不可读取;作者注明"因为之前的 bug,只有改后的性能数据"可用于 Qwen-235B(即无 before/after 同口径对比)。此外 fuseep 对比 deepep 的表格见 PR 截图,文本中无数字。除 10% 外,PR 未注明其他测试条件(硬件型号、并发、序列长度等)。

## 验证与回归风险

精度验证为 Qwen-32B 与 Qwen-235B 的截图(内容不可读,PR 未给文字结论);PR 勾选了单元测试与文档项但素材中未见具体测试文件。风险面:改动面大且跨 NPU 专属路径——CMO 流生命周期改为显式管理影响所有用 CMO prefetch 的模型;`_handle_eplb_and_dispatch` 的条件从 `is not None` 改为 `!= "trivial"`,改变 init_expert_location 语义;`split_qkv_rmsnorm_rope` 与 `get_cos_sin_with_position` 的跨层缓存要求第 0 层先行执行;`topk` 归一化顺序变化(先 l1_norm 后 fp32 cast)可能影响数值路径。CUDA/其他硬件路径均由 `_is_npu`/backend 判断隔离。

## 关联

#10337(动机关联)、#10574([Ascend] optimize Qwen3 on Ascend,PA 进 NPUGraph 的相关工作)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
