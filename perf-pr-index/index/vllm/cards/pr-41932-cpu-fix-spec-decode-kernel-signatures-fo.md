# PR #41932: [CPU] Fix spec decode kernel signatures for synthetic mode compatibility

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jmamou |
| 合并 | 2026-05-10 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready, cpu |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/41932 |

## 一句话结论

补齐 CPU fallback 内核签名修复 spec decode 崩溃，Xeon 8580 / opt-1.3b 吞吐 27.41→60.48 tok/s（2.21x）。

## 动机

PR 正文指出 CPU speculative decoding 在 vLLM v1 下立即崩溃，报 `TypeError: _rejection_greedy_sample_kernel_impl() got an unexpected keyword argument 'SYNTHETIC_MODE'`。根因是 rejection sampler（`vllm/v1/sample/rejection_sampler.py`）为支持 synthetic mode 会传入 `uniform_probs`、`synthetic_conditional_rates`、`SYNTHETIC_MODE`，但 CPU fallback 内核（`vllm/utils/cpu_triton_utils.py`）的签名不接受这些参数。此前 CPU spec decode 完全不可用，值得修复以启用该路径。

## 优化点

在 `_rejection_greedy_sample_kernel_impl` 签名中增加 `uniform_probs=None`、`synthetic_conditional_rates=None`、`SYNTHETIC_MODE=False`，在 `_rejection_random_sample_kernel_impl` 中增加 `synthetic_conditional_rates=None`、`SYNTHETIC_MODE=False`。这些参数被接收但不使用（C++ CPU 内核尚未实现 synthetic mode），仅用于维持与 rejection sampler 调用约定的兼容；两处都加 `assert not SYNTHETIC_MODE` 并补充注释说明。所有参数带默认值、保持向后兼容，GPU 内核已处理这些参数故无 GPU 影响。这是正确性/兼容性修复，使 CPU spec decode 从「完全崩溃」变为可用，从而释放 spec decode 本身的吞吐收益。

## 关键改动

```text
vllm/utils/cpu_triton_utils.py: 两个 CPU Triton fallback 内核签名补齐 synthetic mode 参数、加断言与注释（diff 中仅此 1 个文件）。

  def _rejection_greedy_sample_kernel_impl(
      ...
      max_spec_len,
  +    uniform_probs=None,
  +    synthetic_conditional_rates=None,
  +    SYNTHETIC_MODE=False,
  ):
  +    assert not SYNTHETIC_MODE, "Synthetic acceptance not supported with CPU sampling"
```

## 性能声明

测试条件（Test Plan）：Intel Xeon Platinum 8580（Emerald Rapids，120 核 240 线程），`VLLM_USE_V1=1 VLLM_TARGET_DEVICE=cpu`，target `facebook/opt-1.3b`、draft `facebook/opt-125m`（`num_speculative_tokens=5`），`--max-model-len 512`，10 个推理请求各 50 tokens（共 500 tok）；PR 未注明精度。Before Fix：server 初始化即崩溃（上述 TypeError）。After Fix（opt-1.3b + opt-125m，10.8x model ratio，与无 spec decode 的 baseline 对比）：Throughput 27.41 → 60.48 tok/s（2.21x）、Latency 1.824 → 0.826 s/req（2.21x faster）、Total Time 18.2s(500 tok) → 8.3s(500 tok)（55% reduction）。该 2.21x 为「修复后启用 spec decode」相对「无 spec decode」的收益，修复前无法出数。

## 验证与回归风险

Test Plan 给出 Xeon 8580 上 baseline 与 spec decode 两条 server 启动命令、10 请求 × 50 tokens；Test Result 报告修复后 server 成功初始化并达成 2.21x，checklist 提到以 opt-350m 与 opt-1.3b 模型对在 Xeon 8580 测试。diff 中未见新增单元测试。可能受影响的路径：仅 CPU 路径，GPU 内核不受影响；新增 `assert not SYNTHETIC_MODE` 表示 synthetic mode 在 CPU sampling 下仍显式不支持（会断言失败），在 C++ 内核实现前该路径不可用；参数带默认值保持向后兼容。PR 未注明 CI 回归套件。

## 关联

PR 正文 Related 明确提及：#32662（CPU spec decode 支持的前序 PR，本 PR 建立在其之上）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
