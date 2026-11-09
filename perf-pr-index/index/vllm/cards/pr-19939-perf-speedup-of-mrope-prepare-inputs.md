# PR #19939: [PERF] Speedup of MRoPE prepare inputs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vadiklyutiy |
| 合并 | 2025-06-24 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | ready, v1, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/19939 |

## 一句话结论

MRoPE prepare inputs 改 numpy 直写，`_prepare_inputs` 3.869ms→1.487ms（2.6x），Qwen2.5-VL E2E 高负载约 +1.5%。

## 动机

作者指出 `MRotaryEmbedding.get_next_input_positions_tensor` 耗时较多。#16881 停滞较久，本 PR 结合 #16881 与 #17617 的部分改动，以最小改动获得足够加速：用 numpy 重写该函数。

## 优化点

原实现用 `torch.arange(...).expand(3, -1)` 构造每个 completion 段的 MRoPE 位置张量，再赋给 `self.mrope_positions_cpu[:, dst_start:dst_end]`。新实现把 `get_next_input_positions_tensor` 改为接收预分配 numpy 视图 `out` 与 `out_offset`，用 `np.arange(..., dtype=out.dtype)` 生成值并直接写入 `out[:, out_offset:out_offset+num_new_tokens]`；`gpu_model_runner.py` 在 `__init__` 里缓存 `self.mrope_positions_np = self.mrope_positions_cpu.numpy()`，`_calc_mrope_positions` 直接传入该视图与 `dst_start`，不再构造临时 torch 张量再赋值。改动复用已有的 `mrope_positions_cpu` 存储，只在 `__init__` 多缓存一次 `.numpy()` 视图；`_calc_mrope_positions` 的循环、`dst_start`/`dst_end` 切分与 `mrope_pos_ptr` 累加均保持原样，仅把原来先构造 `torch` 张量再赋值给 CPU 张量的两步合成一次 numpy 就地写入，从而省去每 completion 段的 `torch.arange` 与 `.expand` 开销。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: 缓存 mrope_positions_np（CPU 张量的 numpy 视图），_calc_mrope_positions 改为传 out/out_offset 调用。
vllm/model_executor/layers/rotary_embedding.py: get_next_input_positions_tensor 改为 numpy 就地写入，移除 torch.arange().expand。
```

```python
@staticmethod
def get_next_input_positions_tensor(out: np.ndarray, out_offset: int,
                                    mrope_position_delta: int,
                                    context_len: int, num_new_tokens: int):
    values = np.arange(mrope_position_delta + context_len,
                       mrope_position_delta + context_len + num_new_tokens,
                       dtype=out.dtype)
    out[:, out_offset:out_offset + num_new_tokens] = values
```

## 性能声明

测试条件：`vllm serve Qwen/Qwen2.5-VL-3B-Instruct --disable-log-requests --no-enable-prefix-caching`，用 `fib benchmark -rps 50 --input-token-distribution uniform 250 300 --output-token-distribution uniform 150 250 --num-of-imgs-per-req 1 --img-ratios-per-req 512x512 -n 1000 --base-url http://localhost:8000 --endpoint v1/chat/completions --backend openai-chat` 造负载（每秒 50 请求、每请求 1 张 512x512 图像）。用 nvtx 装饰 `GPUModelRunner._prepare_inputs` 测时：Before 3.869 ms，本 PR 1.487 ms（speedup 2.6x），另注 #17617 为 2.511 ms。正文称 Qwen2.5-VL-3B-Instruct 高负载下 E2E 提升约 1.5%。

## 验证与回归风险

PR 正文列出了 Essential Elements 检查表（Purpose / test plan / test results）并给出上述 benchmark 命令与结果，但未给出单测命令、测试通过数、lint / pre-commit 输出或正确性回归数据，PR 未注明相关测试。可能影响路径：使用 MRoPE 位置计算的 Qwen2.5-VL 等多模态模型（labels 含 qwen）的 `_prepare_inputs`；改动限定在 CPU 侧 numpy 写入，GPU/精度路径未变，也未引入新的自定义 kernel 或设备分支。

## 关联

#16881（部分改动来源，曾停滞）、#17617（部分改动来源，_prepare_inputs 2.511 ms 对照）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
