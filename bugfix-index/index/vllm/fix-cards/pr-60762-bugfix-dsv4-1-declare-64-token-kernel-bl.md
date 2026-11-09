# PR #60762: [Bugfix][DSv4.1] Declare 64-token kernel blocks for SM120 sparse backends

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60762](https://github.com/vllm-project/vllm/pull/60762) |
| 作者 | xzwgit |
| 合并日期 | 2026-11-05 |
| labels | bug, documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, torch.compile, ci/build, multi-modality, tool-calling, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, mrv2, mistral, DSv4, dflash, rust, kimi, k3, scheduler, kv-cache-manager, glm, minimax, cohere, DSv4.1, mooncake, pooling, beam, hisparse, recipe, nixl |
| 热度 | 👍 2 · 💬 60 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
SM120 上 DSv4.1 sparse 后端只声明 128-token kernel 块,--block-size 64 协商即报错;改为 SM12x 声明 [64]。

## 问题现象
SM120 上以 `--block-size 64` 服务 DSv4.1 时,block-size 协商阶段直接失败:

```
ValueError: No common block size for 64 (FLASHINFER_MLA_SPARSE_DSV41: [128]; DEEPSEEK_V41_INDEXER: [128])
```

验证栈:vLLM `0.31.1rc1.dev158+gab905a885` + FlashInfer `0.7.1rc5`,8x RTX PRO 6000 (SM120) TP8,`--block-size 64 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}'`。修复后(均匀 `[64]` 与 per-layer 变体)四种补丁形态全部正常服务(17x19=323),GSM8K n=100 89-91%;最终形态于 vLLM 0.33.0 复验:READY 约 7.5 分钟(冷 JIT),KV cache 2,397,337 tokens @ 32,768 max-model-len,4/4 探针正确且 3 次 greedy 输出逐字节一致。

## 根因
正文披露:SM120 上 DSV41 sparse-MLA 相关后端的 `get_supported_kernel_block_sizes` 只返回 `[64 if family(90) else 128]` / `[128]`,而 SM12x kernel 只实例化 64-token sparse-MLA 块(128 触发 block_kv assert),于是与 `--block-size 64` 协商不出公共块大小。

## 修复方案
在两处声明点为 SM12x 增加 `[64]` 分支,镜像既有 SM90 分支;SWA/backend 选择不动:

```python
# vllm/v1/attention/backends/mla/indexer.py
@staticmethod
def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
    # SM12x kernels only instantiate 64-token sparse-MLA blocks (128
    # trips the block_kv assert, #59203).
    if current_platform.is_device_capability_family(120):
        return [64]
    return [64 if current_platform.is_device_capability_family(90) else 128]
```

`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` 的 FlashInfer sparse backend 同样加 `is_device_capability_family(120)` 分支。依赖要求:`flashinfer-python >= 0.7.1rc5`(runtime page sizes);pin `0.7.0.post1` + 本 PR 不能服务,约 110s 后报 `page_block_size=32 is unsupported; decode kernels are instantiated only for page_block_size=64`。无单元测试(评审决定移除会漂移的 pin 值测试),每个 SM12x `[64]` 带指认 #59203 的行内注释。

## 影响与回归风险
- 影响窗口:正文未给出受影响提交范围(验证栈为 vLLM 0.31.1rc1.dev158 与 0.33.0)。
- 回归面:仅 `get_supported_kernel_block_sizes` 的 SM12x 分支;`--block-size 128` 在该栈上两种声明形式仍不能服务(A 形式:BLHNC layout guard;C 形式:FlashInfer `block_kv == 32 or block_kv == 64`),本 PR 只覆盖默认/64——默认 BLHNC 布局下唯一可服务的几何;四个可服务配置 decode 吞吐差 <1%。非 SM12x 平台路径不变。

## 关联
- #59203(SM120 服务失败;方案出自 @aagallo 在其中的描述)
- #57292(FlashInfer 缺 runtime page size 时 fail-fast,建议配套落地)
- #60948(FlashInfer bump,评审定的落地顺序在其之后)
- #56509(同一失败的重叠声明,已协调)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
