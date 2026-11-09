# PR #38461: [Perf] [NPU] attn: add not needs_cpu_seq_lens for dsa modules

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @litmei |
| 合并 | 2026-10-20 |
| 分类 | Model-specific Fast Path | tier: major |
| Label | npu, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 4 个文件:`environ.py`、`ascend_backend.py`、`eagle_draft_npu_graph_runner.py`、新增 manual 测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/38461 |

## 一句话结论

Ascend NPU attention 后端对 DSA 模型关闭 needs_cpu_seq_lens,消除 CPU 侧同步气泡,GLM5.2 W4A8 TP16 DP4 提速 1.30%。

## 动机

`AscendAttnBackend` 继承 `needs_cpu_seq_lens=True`,即每个 forward 都要取 `seq_lens_cpu` 并做 host 侧同步(`.max().item()` 等),形成 CPU-bound 气泡。但对并不真正需要 `seq_lens_cpu` 的 DSA 模型,这条依赖可以绕过。本 PR 在运行 DSA 模型时把 `needs_cpu_seq_lens` 设为 False,让设备侧张量直接参与计算。前序 #21895 追求同一目标(消除 Draft 执行中的 CPU-bound overhead、同样只适用 DSA 模型),但作者指出直接设 `needs_cpu_seq_lens=False` 更优:不引入额外计算,保持原执行流程。(正文另提及 #36472 但已划线删除。)

## 优化点

`AscendAttnBackend` 与 `AscendAttnMultiStepDraftBackend` 新增 `needs_cpu_seq_lens` 成员:运行无 HybridSWA 的 DSA 模型时置 False(由新环境变量 `SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS` 控制,默认 True,置 0 时 assert 非 hybrid_swa)。`init_forward_metadata` 与 `_apply_cuda_graph_metadata` 增加对应分支:block table 不再按 host 计算的 `seq_lens_max` 截断,直接对 `req_to_token` 做 page-stride 全宽切片;跳过 `seq_lens_cpu_int`、host 侧 `attention_kv_lens` 计算与 `max_len .item()` 等同步点。另外把 `EAGLEDraftNpuGraphRunner.can_run_graph` 的 all_reduce 从 device group 移到 CPU group(decision tensor 放 CPU),消除 draft 图前的阻塞等待。新增 manual 测试 `test_npu_glm_5_1_w4a8_draft_prefetch.py`,以 `SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0` 验证 GLM-5.1-W4A8 + NextN + dp-attention 的精度。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py: needs_cpu_seq_lens=False 分支(init_forward_metadata / _apply_cuda_graph_metadata / MultiStepDraftBackend)
python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py: can_run_graph 的 all_reduce 改走 CPU group
python/sglang/srt/environ.py: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS 开关(默认 True)
test/manual/ascend/llm_models/test_npu_glm_5_1_w4a8_draft_prefetch.py: 新增 manual 精度测试
```

```python
if not envs.SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS.get():
    assert not self.is_hybrid_swa, (
        "Hybrid swa is not supply with `needs_cpu_seq_lens=False`."
    )
    self.needs_cpu_seq_lens = False
```

## 性能声明

条件:Ascend NPU A3、GLM-5.2-0610-Provider-w4a8(modelslim 量化)、NextN spec(steps 2 / topk 1 / draft tokens 3)、deepep auto、`--disable-radix-cache`;bench 为 `bench_serving` random 100 输入/1000 输出。

- TP16 DP1(dp-attn),(1k input/1k output) x 32 concurrency x 4 次:预期收益 **0.79%**(bubble 0.68 ms,有效计算+通信 85.82 ms);base duration 159.35/157.87/156.99 s,feat 159.64/156.81/155.81 s,TPOT 约 33.2-33.9 ms,acc len 约 4.67-4.69。
- TP16 DP4,(1k/1k) x 32 concur:预期收益 **1.30%**(bubble 1.07 ms,有效 80.93 ms);base 9 次 duration 116.00-121.52 s(TPOT 24.05-25.38),feat 10 次 113.84-117.14 s(TPOT 23.50-24.31)。
- 准确性:GSM8K full(1319)accuracy 0.951(`few_shot_gsm8k`,2-shot,parallel 32)。

## 验证与回归风险

测试:GSM8K 1319 题全量 0.951;新增 manual 测试覆盖 GLM-5.1-W4A8 + draft prefetch + npu graph + NextN 组合(env 打开本特性)。风险:仅对 DSA 且非 HybridSWA 模型生效,HybridSWA 下显式 assert 拒绝;False 分支的 block table 按 `req_to_token` 全宽构建,graph capture 的 block-table 宽度语义与 host 截断路径不同;CPU group all_reduce 依赖 `tp_group.cpu_group` 存在;checklist 各项未勾选;默认关闭(env 置 0 才生效),不开时零变化。

## 关联

#21895(前序:同目标——消除 Draft 执行 CPU-bound overhead,仅 DSA 模型,方案不同);#36472(正文提及后划线删除,未采用为依赖)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
