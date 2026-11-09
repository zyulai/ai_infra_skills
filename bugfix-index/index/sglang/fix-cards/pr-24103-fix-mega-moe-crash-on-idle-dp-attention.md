# PR #24103: Fix mega-MoE crash on idle DP-attention ranks under tvm-ffi DeepGEMM

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24103](https://github.com/sgl-project/sglang/pull/24103) |
| 作者 | Fridge003 |
| 合并日期 | 2026-04-29 |
| labels | deepseek |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
空闲 DP-attention rank 上零行输出张量 data_ptr 为空,tvm-ffi 绑定拒绝 nullptr 崩溃;改为至少分配 1 行。

## 问题现象
使用 tvm-ffi DeepGEMM 的 mega-MoE 路径下,空闲的 DP-attention rank(num_tokens == 0)会导致 scheduler 崩溃。发送任意请求即触发,报错为 `tvm.error.InternalError: The specified pointer resides on host memory and is not registered with any CUDA device.`,起始帧为 `deep_gemm::convert_to_torch_tensor` → `from_blob` → `getDeviceFromPtr`。正文给出完整复现命令:DeepSeek-V4-Flash,`CUDA_VISIBLE_DEVICES=4,5,6,7 NVSHMEM_DISABLE_IB=1`,开启 `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` 等环境变量,`--tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend deepep`。

## 根因
正文披露了完整根因链:
1. 空闲 DP-attention rank 上(num_tokens == 0),`_run_mega_routed` 分配 `y = torch.empty((0, H), ...)`,CUDA storage 为空,`data_ptr()` 返回 `nullptr`;
2. DeepGEMM 的 tvm-ffi 绑定(上游 commit `ffb3d29`,"support tvm ffi interfaces")通过 `from_blob(data_ptr, ...)` 从 `tvm::ffi::TensorView` 重建 `torch::Tensor`,其间调用 `getDeviceFromPtr(data_ptr)`,对 nullptr 直接拒绝并抛出上述 InternalError;
3. mega-MoE 是跑在 EP symmetric buffer 上的集合操作,rank 无法跳过这次调用,因此空闲 rank 必然崩溃。

## 修复方案
改动 `python/sglang/srt/models/deepseek_v2.py` 的 `_run_mega_routed`:

```python
# Allocate at least one row so y has a non-null CUDA data_ptr;
# the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
y = torch.empty(
    (max(num_tokens, 1), hidden_size),
    dtype=torch.bfloat16,
    device=hidden_states.device,
)
...
y = y[:num_tokens]
```

即保证 `y` 至少有 1 行以获得合法 CUDA 指针,DeepGEMM 调用完成后、在 scaling 等后续处理前截断回 `num_tokens`,调用方可见行为不变。

## 影响与回归风险
- 影响窗口:正文未说明(自上游 tvm-ffi 绑定 commit `ffb3d29` 引入后)。
- 回归面:仅影响 `_run_mega_routed` 的输出张量分配与截断;`num_tokens == 0` 时会真实分配 1 行的临时显存(开销极小),`num_tokens > 0` 时行为不变。修复后 `y[:num_tokens]` 依赖 DeepGEMM 只写前 num_tokens 行,若 kernel 写满整个 1 行缓冲,截断后语义仍一致,但属实现细节假设。

## 关联
正文未列出(未附 issue/PR 链接;正文提及上游 commit `ffb3d29` 与 DeepGEMM `2e21741`)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
