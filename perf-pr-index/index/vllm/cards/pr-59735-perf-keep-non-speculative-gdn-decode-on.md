# PR #59735: [Perf] Keep non-speculative GDN decode on the standard path

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @DimensionSTP |
| 合并 | 2026-10-02 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动为 1 个 GDN 模型文件 + 1 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/59735 |

## 一句话结论

未配置推测解码的 GDN 模型不再走 packed fused-normalization 包装,回到标准 recurrent+normalization 路径,H100 上 Qwen3.5-9B 吞吐 +1.31%/+1.47%。

## 动机

融合 CUDA recurrent 算子是为 speculative/MTP batch 设计的:外层 forward 路径此前仅按配置的 CUDA backend(`VLLM_GDN_DECODE_KERNEL=cuda`)选择 packed wrapper,即使模型根本没有配置推测解码、不可能产生 MTP batch,也会走进该 wrapper 及其 fallback normalization。这条对普通模型无用的包装路径带来额外开销(issue #59520 报告在 H200 上有 30-35% 的差距)。

## 优化点

机制是纯 dispatch 修正:构造时新增 `enable_fused_gdn_spec_decode = (gdn_decode_kernel == "cuda" and speculative_config is not None)`,把"能否用融合路径"从仅看 kernel 选择收紧为"kernel 为 CUDA 且确实配置了推测解码",是静态构造期决策,不引入逐 batch 的 host 读取或同步。`forward_cuda` 的 `use_fused_gdn_spec_decode` 判定替换原 `use_fused_gdn_decode`;普通模型保留既有标准 recurrent 与 normalization 路径,配置了推测解码的模型 dispatch 完全不变(含 prefill 与 mixed-batch fallback)。同时 CUDA 不可用时的 fallback 日志只在有 speculative_config 时输出,日志措辞改为 "GDN spec decode kernel"。不新增 kernel、硬件选择启发式、依赖或公开选项。

## 关键改动

```text
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: enable_fused_gdn_decode 改名 enable_fused_gdn_spec_decode 并叠加 speculative_config is not None 条件;forward_cuda 判定随之替换
tests/kernels/mamba/test_gdn_fused_mtp.py: 测试中的 layer stub 标志同步改名
```

关键判定改动:

```python
self.enable_fused_gdn_spec_decode = (
    self.gdn_decode_kernel == "cuda" and self.speculative_config is not None
)
...
use_fused_gdn_spec_decode = (
    self.enable_fused_gdn_spec_decode
    and hidden_states.dtype == torch.bfloat16
    and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
)
```

## 性能声明

Qwen/Qwen3.5-9B(revision `c202236...`),BF16,2x H100,TP=2,Model Runner V2,Torch 2.13.0+cu129;`vllm bench throughput`,prefix_repetition 数据集(20 prefixes x 32768 前缀 + 1024 suffix,输出 1024,200 请求,每臂恰 204,800 输出 tokens),值為 mean output tokens/s:prefix-cache retention `0`:1708.92 → 1731.32(+1.31%);retention `None`:1889.00 → 1916.75(+1.47%)。作者明确说明:issue #59520 报告的 30-35% H200 提速未在本环境复现,H100 收益更小,且不声明跨硬件等价或模型质量等价。

## 验证与回归风险

`tests/kernels/mamba/test_gdn_fused_mtp.py` 16 passed 无 skip(含既有 MTP、普通 decode、prefill、mixed-batch 数值测试,由 Mamba Kernels CI job 覆盖);CPU metadata 与 worker 控制 62 passed、3 个 CUDA-only 测试跳过;真实模型 greedy 对比 8 个 prompt 输出 token ID 全部一致(作者注明这是 output smoke 而非 benchmark 级等价声明);10 个官方 pre-commit hook 通过。注意:吞吐与 smoke 数字采集于标志改名之前,改名后仅复跑了 CUDA 套件与 CPU 控制组,AST 比对确认无额外运行时逻辑变化。风险面:改动只影响 Qwen GDN linear attention 层的外层 dispatch;`VLLM_GDN_DECODE_KERNEL=cuda` 但未配置推测解码的部署从 fused wrapper 切回标准路径,若某部署依赖该 wrapper 的其他副作用(正文未指出)需关注;配置了推测解码的模型行为逐位不变。

## 关联

同区域:issue #59520(动机来源);#53463(为普通 decode 启用 draft width 1 的融合 CUDA recurrent,与本 PR 方向不同)、#41966(FlashInfer GDN decode backend,开发中)、#53645(SM120 专用工作)、#51954(标准 decode 内去除 gate 拷贝);GDN/Mamba 相关见 pr-43534(CPU fused GDN)、pr-61855(ROCm fused GDN MTP triton port)、pr-65874(fused GDN MTP DSpark support)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
