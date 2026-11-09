# PR #24252: [Compile] Conditional compilation. Introduce compile_ranges

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-12-05 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | performance, frontend, ready, torch.compile, ci/build, v1, tool-calling |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 8 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/24252 |

## 一句话结论

把 torch.compile 的按固定 size 编译泛化为按 compile range 编译,支持按输入 batch size 动态分发到不同预编译图,从而让 allreduce 融合只作用于小中形状、大形状回退 NCCL。

## 动机

对应 RFC #23113。此前 vLLM 的 piecewise 编译只支持 `compile_sizes`(对若干固定 batch size 各编译一份特化图)加一份 dynamic shape 图;fusion pass 只能整图开或关,无法按 batch 大小区分。小中 batch 下 fused allreduce(如 FlashInfer/量化融合)更快,而大 batch 下 NCCL allreduce 更优,但 dynamic shape 图内无法静态地按形状选择路径。本 PR 是拆分 #22086 的第二部分(第一部分 #24248),目标是让用户能按 batch 大小粒度地静态启用/禁用融合。

## 优化点

机制上:(1) 新配置 `compile_ranges_split_points`,由 split points 定义区间,例如 (32, 64) 生成 [1,32]、[33,64]、[65,max_num_batched_tokens) 三个 range;`compile_sizes` 泛化为单元素 range 且优先命中。(2) `PiecewiseBackend` 从 `ConcreteSizeEntry` 改为 `RangeEntry`:运行时按 sym shape 找到所属 range,未编译则触发该 range 的编译(非单元素 range 需先 fakify example inputs,避免 torch.compile 对真实张量的意外 specialization);动态 shape 兜底图被移除,所有编译都经由 `piecewise_backend.py` 在某个 range 边界内进行。(3) `CompilerManager`/compiler interface 的缓存 key 与 `dynamic_shapes` 策略改为按 Range(单元素 range 用 from_example_inputs,区间 range 用 from_graph),pass context、pass uuid、`is_applicable(shape)` 全部迁移为 `is_applicable_for_range(Range)`。(4) 开启 allreduce fusion 时,vLLM 自动追加一个 split point 把图分成两份:小中形状用 fused allreduce,大形状用 NCCL allreduce;`is_applicable_for_range` 以 `compile_range.end <= max_token_num` 等条件判定该 range 是否应用融合。

## 关键改动

```text
vllm/compilation/piecewise_backend.py: RangeEntry 按区间编译与动态分发,移除 dynamic shape 兜底图
vllm/compilation/backends.py: CompilerManager 缓存 key 由 runtime_shape 改为 Range
vllm/compilation/compiler_interface.py: standalone_compile 按 Range 选择 dynamic_shapes 策略
vllm/compilation/collective_fusion.py: FlashInfer allreduce 融合按 range 判定适用性
vllm/config/compilation.py: 新增 compile_ranges_split_points 配置
```

关键片段(按 shape 找 range):

```python
def _find_range_for_shape(self, runtime_shape: int) -> Range | None:
    if runtime_shape in self.compile_sizes:
        return self.range_entries[Range(start=runtime_shape, end=runtime_shape)]
    else:
        for range in self.compile_ranges:
            if runtime_shape in range:
                return self.range_entries[range]
    return None
```

## 性能声明

PR 正文给出端到端数字(服务端 `VLLM_ALLREDUCE_USE_SYMM_MEM=1 vllm serve -tp N --max-num-seqs 256`,开启 `enable_fi_allreduce_fusion`;客户端 input 1024 / output 128;硬件 B200)。Baseline vs "Allreduce + RMSNorm + QuantFp8":

- B200 TP=2,Llama-3.1-70B-Instruct-FP8:qps=1 Mean TTFT 85.644→71.489 ms,Mean TPOT 11.812→10.725 ms;qps=10 TPOT 27.632→23.219 ms;qps=15 Mean TTFT 291.392→201.879 ms,Mean TPOT 46.534→36.656 ms。
- B200 TP=4,Qwen3-Next-80B-A3B-Instruct(No EP):qps=15 TPOT 49.314→47.549 ms,TTFT 120.676→116.451 ms(收益较小)。
- B200 TP=8,DeepSeek-V3.1(No EP):qps=10 TPOT 22.646→20.094 ms;qps=15 TPOT 27.765→25.690 ms,TTFT 102.587→97.646 ms。

启动时间代价:两图编译的典型配置下 DeepSeek-V3 冷启动 181.91 s,热启动 12.40 s(PR 明确说明编译图增多会拉长启动)。

## 验证与回归风险

新增 `test_compile_ranges.py`;config hash 因子纳入 compile range(pass uuid 中加入 range 字符串,保证 inductor 对新动态 range 重新编译)。回归风险:shape 不落在 [1, max_num_batched_tokens] 内会直接 assert("Shape out of considered range");fake example inputs 的 duck-shaping 问题(PR 引用 issue #27899)是敏感点;PR 自列 follow-up:多图共享 shape env 可能互相约束 SymInt、融合 pass 拟移至 O3 编译级别、range 信息与 inductor 的 SymInt 共享。

## 关联

系列:#24248(第一部分)、#24604(本 PR 基于其上)、#22086(被拆分的原 PR);RFC issue #23113。索引内同区域(CUDA Graph / Compiler)卡片:#20059(full cudagraph + separate sampler)、#25205(AOT compilation)、#25444(默认 CUDAGraphMode FULL)、#24123(WIDE-EP piecewise cuda graph)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
