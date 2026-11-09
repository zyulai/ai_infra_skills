# PR #20412: [V0 deprecation] Remove V0 CPU/XPU/TPU backends

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2025-07-06 |
| 分类 | Other \| tier: major |
| Label | documentation, performance, structured-output, frontend, tpu, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)，pack 末尾标注 17 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/20412 |

## 一句话结论

V0 弃用第一步：删除已迁移到 V1 的 CPU/XPU/TPU 后端，PR 未给出性能数字。

## 动机

PR 正文仅一句：本 PR 是 V0 deprecation 的一部分；作为开始，将删除 vLLM V0 中已迁移到 vLLM V1 的硬件后端，即 CPU、XPU 和 TPU。除此之外 PR 正文未说明更多背景或瓶颈。

## 优化点

删除 V0 的 CPU/XPU/TPU 后端实现：整文件删除 `vllm/attention/backends/cpu_mla.py`（307 行，CPU MLA backend）、`vllm/attention/backends/ipex_attn.py`（XPU/IPEX attention）、`vllm/attention/backends/pallas.py`（TPU/Pallas attention）等。测试侧 `tests/kernels/attention/test_attention_selector.py` 把 `DEVICE_MLA_BLOCK_SIZES["cpu"]` 由 `[16]` 改为 `[]`（标注 `FIXME(woosuk): Temporarily disable CPU tests`），并在 `test_env` 中对 `device == "cpu"` 且 `not use_v1` 增加 `pytest.skip("CPU backend only supports V1")`。改动以代码删除为主，PR 正文未说明更多机制。

## 关键改动

```text
vllm/attention/backends/cpu_mla.py: 整文件删除（307 行，CPU MLA backend）。
vllm/attention/backends/ipex_attn.py: V0 XPU/IPEX attention backend 删除。
vllm/attention/backends/pallas.py: V0 TPU/Pallas attention backend 删除。
tests/kernels/attention/test_attention_selector.py: 禁用 V0 CPU 用例、仅 V1 生效。

  DEVICE_MLA_BLOCK_SIZES = {
      "cuda": [16, 64],  # CUDA supports both standard and extended block sizes
      "hip": [16, 1],  # HIP requires special handling for block_size=1
  -    "cpu": [16]  # CPU uses fixed block size from test cases
  +    # "cpu": [16]  # CPU uses fixed block size from test cases
  +    "cpu": []  # FIXME(woosuk): Temporarily disable CPU tests
  }
```

pack 的 diff 仅含 4 个文件，末尾标注 17 file(s) dropped from diff for size，其余被删除/改动的文件未在 pack 中展开。

## 性能声明

PR 正文未给出定量性能数据（regex 未检测到 perf claims），也未提供 benchmark、测试条件、模型或硬件信息；本 PR 为 V0 弃用过程中的代码删除，非性能优化改动。

## 验证与回归风险

`tests/kernels/attention/test_attention_selector.py`：`DEVICE_MLA_BLOCK_SIZES["cpu"]` 置空并标注 FIXME 临时禁用 CPU 测试；`test_env` 对 `device == "cpu"` 且 `not use_v1` 时 `pytest.skip("CPU backend only supports V1")`。风险与边界：删除 V0 的 CPU/XPU/TPU 后端后，这些硬件上的 V0 路径不再可用，需改用 V1；CPU MLA 相关 attention selector 测试被临时禁用（FIXME）；pack 有 17 个文件因体积被 dropped，PR 正文未列出完整改动清单，也未给出其他回归测试信息。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
