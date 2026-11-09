# PR #50932: buffer size insuffient Dspark sd for FlashInfer MNNVL allreduce

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50932](https://github.com/vllm-project/vllm/pull/50932) |
| 作者 | khushali9 |
| 合并日期 | 2026-08-27 |
| labels | bug, ready, nvidia, dflash |
| 热度 | 👍 2 · 💬 30 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
FlashInfer MNNVL allreduce 的 token 门控按整块 2MB workspace 计算,忽略 Lamport 三缓冲轮转,49-146 token 的张量过闸后在 kernel 内 abort;改问 workspace 自身容量。

## 问题现象
修复 #50877。`FlashInferAllReduce.should_use_fi_ar` 用 `max_workspace_size // (hidden_dim * element_size)` 计算门控,而 `max_workspace_size` 是整个 MNNVL 分配(TP8 时 2MB);MNNVL 后端基于 Lamport、轮转使用 NUM_LAMPORT_BUFFERS=3 个缓冲,单次 all-reduce 只能用到约 1/3 预算:预算 2,097,152 字节→门控放行 146 token,而单缓冲 698,368 字节→真实容量 48 token。任何 49-146 token 的张量都能过闸,然后在 kernel 内 abort("The buffer size in the given workspace is insufficient")。drafter 每步产生 B×(1+num_speculative_tokens) 个 token——如 14 请求×4=56 token——恰好落进这个窗口;无 drafter 时 decode 批只产生 B 个 token,低于 48 不会触发。

## 根因
正文披露:门控复用整个 workspace 预算做 token 上限,未考虑 mnnvl 后端 Lamport 缓冲轮转导致单次调用只占约 1/3 容量;`fused_allreduce_gemma_rms_norm._can_use_flashinfer` 存在同样缺陷(其 docstring 自述 "mirrors FlashInferAllReduce")。

## 修复方案
保留原有预算上界作为廉价预过滤,之后改为调用 workspace 抽象基类自带的 `is_buffer_size_sufficient(tp_size, num_tokens, hidden_dim, dtype)`(pinned flashinfer-python==0.6.15.post1),不重新实现 Lamport 算术,对 mnnvl 与 trtllm 都正确且版本兼容:
- `flashinfer_allreduce.py` 的 `should_use_fi_ar`:`_ensure_workspace` 成功后取 workspace 并返回 `workspace.is_buffer_size_sufficient(...)`。
- `fused_allreduce_gemma_rms_norm.py` 的 `_can_use_flashinfer`:同样追加该检查。
正文明确未修的部分:`AllReduceFusionPass` 有同样的过松上界,但其 `max_token_num` 同时驱动 workspace 分配请求,修复需拆分这两个关注点并验证 one-shot/two-shot 划分,需 GB200 硬件,留待后续。无代码 workaround:`VLLM_ALLREDUCE_USE_FLASHINFER=0`(不影响 compiled fusion pass)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围;触发条件是 MNNVL(GB200/GB300)+ FlashInfer allreduce + spec decode(dspark drafter)使 token 数落在 49-146。
- 回归面:eager FlashInferAllReduce 路径与 fused_allreduce_gemma_rms_norm 的可用性判定——超出单缓冲容量的张量现在改走非 FlashInfer 回退而非崩溃;AllReduceFusionPass 未动,同样输入经 compiled pass 可能仍有问题。GB300 由 @janbernloehr 验证;作者无 GB200,报文注明该平台验证待完成。

## 关联
- #50877(被修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
