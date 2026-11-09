# PR #43107: [Core] Check for GPU<->CPU syncs during CI

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2026-08-15 |
| 分类 | Other(tier: major) |
| Label | rocm, speculative-decoding, ready, ci/build, v1, multi-modality, qwen, kv-connector, nvidia, ready-run-all-tests, quantization, mistral, cohere |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 28 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/43107 |

## 一句话结论

引入 `VLLM_GPU_SYNC_CHECK` 开关(基于 torch.cuda.set_sync_debug_mode)在 CI 中检测模型 forward/sampler 的 GPU-CPU 同步,并用 `gpu_sync_allowed()` 显式标注数十处已知合理同步,守护异步调度路径不被隐性 sync 拖慢。

## 动机

vLLM 已默认(且绝大多数场景)使用异步调度,主 CUDA stream 上的任何 GPU-CPU 同步都会打断流水线、直接损害性能,而这类 sync 往往不透明、容易在迭代中无意引入(如 `.item()`、`.tolist()`、布尔掩码索引、host 侧策略读回等)。此前 PR #40561 尝试系统性暴露这些 sync;本 PR 在其低垂果实修复合入 main 后 rebase 重做,把"检测机制 + 已知 sync 白名单"固化为可持续的 CI 手段。

## 优化点

机制上两部分:(1) 检测:新增 `VLLM_GPU_SYNC_CHECK` 环境变量,开启后对模型 forward pass 与 sampler 设置 `torch.cuda.set_sync_debug_mode`,同步操作会立即报错而非静默阻塞,便于 CI 捕获回归;新增 `gpu_sync_allowed()` 上下文管理器,把"临时或不可避免"的已知 sync 显式包起来,使其在检查模式下放行且留下注释说明原因。(2) 治理:diff 中对大量现存 sync 点加白名单并注明理由,包括 EPLB 的专家统计 tolist 与 host 侧 rebalance 策略、NIXL connector 的 h2d/d2h 块拷贝、LoRA adapter 加载与 punica metadata 准备(留 TODO 待消除)、Mamba mixer 的 per-sequence Python 标量读取、多个多模态模型(audioflamingo3、cohere_asr、ernie45_vl、gemma4_mm、granite_speech、idefics2/3 等)中布尔掩码索引与 `.tolist()`、测试代码中的断言读回等。

## 关键改动

```text
vllm/utils/gpu_sync_debug.py: 新增 gpu_sync_allowed() 上下文管理器与 sync 检测设施
vllm/distributed/eplb/*: 专家统计/rebalance/权重重排中的同步点加白名单
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py: h2d/d2h 块拷贝标注为有意同步
vllm/lora/punica_wrapper/punica_gpu.py / worker_manager.py: LoRA metadata 与 adapter 加载同步点
vllm/model_executor/layers/mamba/mamba_mixer2.py: conv_ssm_forward 的 per-sequence 标量读回合并为一次 tolist
```

关键片段(把循环内多次隐式 sync 收敛为一次显式读回):

```python
with gpu_sync_allowed():
    block_idx_first_cpu = block_idx_first_scheduled_token_p.tolist()
    block_idx_last_cpu = block_idx_last_scheduled_token_p.tolist()
    num_computed_tokens_p_cpu = num_computed_tokens_p.tolist()
    last_chunk_indices_p_cpu = last_chunk_indices_p.tolist()
```

## 性能声明

PR 正文未提供 benchmark 数字(此 PR 为性能基础设施/防回归机制,不直接声称性能收益,perf claims 检测为 none)。其价值在于消除异步调度路径上的隐性 GPU-CPU 同步。

## 验证与回归风险

依赖 PyTorch 的 `torch.cuda.set_sync_debug_mode`;测试侧在 `tests/models/multimodal/generation/test_mm_prefix_lm.py`、`tests/v1/e2e/general/test_mamba_prefix_cache.py` 等处把断言用读回包进 `gpu_sync_allowed()` 并注明 deliberate D2H。回归风险:开启 `VLLM_GPU_SYNC_CHECK` 的 CI 会对新增未标注 sync 报错,可能需要随模型/连接器迭代持续维护白名单;被包裹的同步点本身行为不变(LoRA metadata 处留有 TODO 待真正消除 sync)。

## 关联

取代 PR #40561(前序版本,rebase 后重做)。与 #27614(默认启用异步调度)直接配套——异步调度正是本机制守护的性能前提。索引内同区域/Core 相关卡片:#21917(scheduler request removal 优化)、#21146(cudagraph capture 期间冻结 GC)、#24128(cudagraph 后运行 GC)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
