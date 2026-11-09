# PR #18437: [KERNEL] Sampler. CUDA kernel for applying repetition penalty

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vadiklyutiy |
| 合并 | 2025-06-04 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, ci/build |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 6 个文件（新增 sampler.cu 与测试，改 CMakeLists、ops.h、torch_bindings.cpp、_custom_ops.py、layers/utils.py） |
| 链接 | https://github.com/vllm-project/vllm/pull/18437 |

## 一句话结论

新增单遍 CUDA kernel 原地应用 repetition penalty，替代 Sampler 中的多次 torch 算子组合，H100 上微基准提速 2.85–6.23 倍，小模型 batch=256 E2E 平均延迟降 5.9%。

## 动机

Sampler 中施加 repetition penalty 的 torch 实现（unsqueeze/repeat、两次 `torch.where`、逐元素乘）由多个算子组成，对 [num_seqs, vocab] 全量张量反复读写，小模型场景下该部分在 Sampler 中占比明显。作者称"尤其对 small models 耗时长"。

## 优化点

新增 `csrc/sampler.cu`：一个 kernel 完成"查 mask → 判断正负 logit → 除/乘 penalty"的完整逻辑，单次读 logits + masks、仅对重复 token 写回，in-place 更新。grid 为 `(num_seqs, tile_num)`，每 block 处理一个 sequence 的一段 vocab tile；`tile_num` 按 SM 数与 num_seqs 拿捏（`min(vocab_size, max(1, (sms+num_seqs-1)/num_seqs))`），`block` 尺寸取 `min(tile_size, 1024)`，线程在 tile 内跨步循环。dispatch 用 `VLLM_DISPATCH_FLOATING_TYPES` 支持 fp32/fp16。Python 侧保留 torch 参考实现 `apply_repetition_penalties_torch` 作为非 CUDA/非连续输入的 fallback，统一入口 `apply_repetition_penalties` 按 `current_platform.is_cuda() and logits.is_contiguous()` 分派；`layers/utils.py` 的 `apply_penalties` 改调该 custom op，删除原来的内联 torch 逻辑。ROCm 走 hipcub include 分支（kernel 本体未用 cub，仅头文件按平台切换）。

## 关键改动

```text
csrc/sampler.cu: 新增，apply_repetition_penalties_kernel（按 seq×tile 网格、单遍 in-place 施加 penalty）
vllm/_custom_ops.py: 新增 apply_repetition_penalties 统一入口 + torch fallback 参考实现
vllm/model_executor/layers/utils.py: apply_penalties 中 repetition penalty 部分改调 custom op
csrc/torch_bindings.cpp / ops.h / CMakeLists.txt: 注册 apply_repetition_penalties_ 算子并纳入构建
tests/kernels/test_apply_repetition_penalties.py: 新增，num_seqs×vocab×dtype 参数化对拍 + opcheck
```

kernel 核心：

```cpp
const bool is_repeated = prompt_mask[idx] || output_mask[idx];
if (is_repeated) {
  scalar_t logit = logits[idx];
  if (logit > 0) {
    logits[idx] = logit / penalty;
  } else {
    logits[idx] = logit * penalty;
  }
}
```

## 性能声明

微基准（H100，vocab 151936，penalty 1.2，CUDA event 计时，20 次迭代含 `logits.clone()`）：torch vs CUDA kernel 提速 **2.85x–6.23x**——num_seqs=1: 6.23x（0.0753→0.0121 ms）、8: 5.78x、16: 3.68x、32: 2.85x、64: 3.40x、100: 3.42x、256: 3.85x、1024: 3.85x、1025: 3.85x；与 torch 实现最大差 2.4e-7（fp32）。E2E（`benchmark_latency.py`，Llama-3.2-1B-Instruct、input 256/output 256、batch 256、repetition_penalty 1.05、H100）：平均延迟 2.697 s → 2.546 s，**Avg latency speed up 5.9%**。

## 验证与回归风险

新增 `tests/kernels/test_apply_repetition_penalties.py`：num_seqs 1–1025、vocab 17/256/1019/151936/202048、fp32/fp16 对拍（rtol/atol 1e-3）+ `opcheck` 算子自检。风险面：(1) 分派条件要求 `logits.is_contiguous()`，否则退回 torch 路径（结果一致但无加速）；(2) kernel 依赖 prompt/output mask 预先算好（`get_token_bin_counts_and_mask` 逻辑不变），若 mask 构建本身是瓶颈则收益有限；(3) 语义与 torch 参考严格一致（正 logit 除、负 logit 乘），对拍测试锁定。

## 关联

Sampler/sampling kernel 同区域：素材未列出其他编号。测试文件名与 CI kernels 列表中的 `kernels/test_apply_repetition_penalties.py`（见 PR #61895 diff 上下文中的 CI yaml）对应，说明该测试后来进入常规 CI 分片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
