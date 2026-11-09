# PR #12093: [Kernel] Flash Attention 3 Support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2025-01-23 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ready, ci/build |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/12093 |

## 一句话结论

为 Hopper 引入 FlashAttention 3 后端并在 SM90+ 上默认启用,统一 FA2/FA3 在 paged KV cache 下的接口(改用 seqused_k),V1 吞吐提升约 11%。

## 动机

FA3 的大部分改动在 vllm-flash-attn(fork 仓库)内完成,但 vLLM 侧仍需适配:FA3 在使用 paged KV cache 时不支持 `cu_seqlens_k`,而是使用 `seqused_k`(即 KV seqlens)。由于 FA2 也支持 `seqused_k`,本 PR 把 FA2/FA3 在 paged KV cache 场景下统一到 `seqused_k` 接口,以维护一个共同接口。目标是在 Hopper(H100 等)上获得 FA3 的 decode/prefill 性能收益,当前主要改善 V1 引擎性能。

## 优化点

机制上:(1) vllm-flash-attn fork 升级到支持 FA3 的 commit,CMake/ setup.py 把单一 `vllm_flash_attn_c` 组件拆成 `_vllm_fa2_C` 与 `_vllm_fa3_C` 两个扩展,FA2/FA3 可分别构建安装;(2) `FlashAttentionBackend`(V0 与 V1 两份)在初始化时按设备算力选版本:compute capability >= 9 默认 FA3(若 fork 支持),否则 FA2;环境变量 `VLLM_FLASH_ATTN_VERSION=[2|3]` 可强制指定;(3) 接口统一:paged KV cache 路径下 prefill/decode/cascade attention 全部从 `cu_seqlens_k` 换成 `seqused_k`,cascade attention 的 prefix/suffix kv lens 也从 cu 格式改为普通长度数组,相应地 V1 `gpu_model_runner._prepare_inputs` 不再构造 cumsum 的 `seq_start_loc`,直接维护 `seq_lens` 张量,减少一次前缀和计算;(4) 所有 `flash_attn_varlen_func`/`flash_attn_with_kvcache` 调用透传 `fa_version` 参数。

## 关键改动

```text
vllm/attention/backends/flash_attn.py: V0 后端版本选择(VLLM_FLASH_ATTN_VERSION / SM90 默认 FA3)与 seqused_k 接口切换
vllm/v1/attention/backends/flash_attn.py: V1 后端同样切换,cascade attention 改 prefix/suffix_kv_lens
vllm/v1/worker/gpu_model_runner.py: _prepare_inputs 维护 seq_lens 取代 seq_start_loc cumsum
CMakeLists.txt / setup.py: vllm-flash-attn 拆为 _vllm_fa2_C/_vllm_fa3_C 组件
tests/kernels/test_flash_attn.py, tests/kernels/test_cascade_flash_attn.py: 参数化 fa_version=2/3
```

关键片段(版本选择):

```python
# if hopper default to FA3, otherwise stick to FA2 for now
if current_platform.get_device_capability()[0] >= 9:
    self.fa_version = 3 if is_fa_version_supported(3) else 2
else:
    self.fa_version = 2
if VLLM_FLASH_ATTN_VERSION is not None:
    assert VLLM_FLASH_ATTN_VERSION in [2, 3]
    self.fa_version = VLLM_FLASH_ATTN_VERSION
```

## 性能声明

PR 正文给出 benchmark_throughput 数字(模型 meta-llama/Llama-3.1-8B-Instruct,input-len 2048,output-len 128,max-num-batched-tokens 16384;硬件未注明,推断为 Hopper 但 PR 未写明):

- V1(VLLM_USE_V1=1):FA2 12.81 req/s / 27872.21 total tok/s / 1639.54 output tok/s;FA3 14.24 req/s / 30986.15 total tok/s / 1822.71 output tok/s(约 +11% 吞吐)。
- V0:FA2 12.40 / 26991.51 / 1587.74;FA3 12.74 / 27714.72 / 1630.28。
- V0 + num-scheduler-steps 4:FA2 13.20 / 28726.44 / 1689.79;FA3 13.34 / 29029.89 / 1707.64。

## 验证与回归风险

`tests/kernels/test_flash_attn.py` 与 `tests/kernels/test_cascade_flash_attn.py` 为现有 kernel 测试增加 `fa_version=2/3` 参数化;FA3 在 compute capability 8.6/8.9 上因部分形状 shared memory 不足被 skip。回归风险:接口从 `cu_seqlens_k` 改为 `seqused_k` 影响 FA2 与 FA3 的全部 paged KV 调用路径(V0/V1、cascade);FA3 仅默认用于 SM90+,Ampere 等旧架构保持 FA2;FA3 在 Ampere 上的收益未评估(PR 内 TODO 注明待 profile)。

## 关联

前置:vllm-flash-attn fork(https://github.com/vllm-project/flash-attention)中的 FA3 实现;后续 vLLM 内 FA3 相关优化(编号素材未注明)。索引内同区域卡片:#14152(V1 triton/GPU model runner 优化)、#20034(FlashInfer ragged prefill)、#21078(FlashInfer MLA decode)、#13747(FlashMLA)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
