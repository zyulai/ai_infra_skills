# PR #20516: [Benchmark] Add support for multiple batch size benchmark through CLI in `benchmark_moe.py` + Add Triton Fused MoE kernel config for FP8 E=16 on B200

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @b8zhong |
| 合并 | 2025-07-06 |
| 分类 | MoE |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 含 2 个文件(benchmark 脚本 + 1 个新 JSON config) |
| 链接 | https://github.com/vllm-project/vllm/pull/20516 |

## 一句话结论

让 `benchmark_moe.py` 的 `--batch-size` 支持一次传多个 batch size,并为 FP8 E=16 的 MoE 在 B200 上落盘一份调优好的 Triton fused MoE config。

## 动机

此前 `benchmark_moe.py` 的 `--batch-size` 只接受单个整数,要为一组 batch size 各自调优 fused MoE kernel config,必须逐个 batch size 手动运行调优脚本。对 Llama-4-Scout 这类 16 专家(E=16)的小专家数 MoE,在 B200 上也没有现成的 fp8_w8a8 调优配置,运行时会用不到针对该形状调优的 Triton tile 参数。

## 优化点

改动分两部分。其一,`benchmarks/kernels/benchmark_moe.py` 的 argparse 将 `--batch-size` 改为 `nargs="+"`,接受列表;`main()` 中非默认分支直接使用 `args.batch_size` 列表作为待跑 batch sizes,从而一次运行即可对多个 batch size 完成调优并各自保存 config 文件。其二,新增 `E=16,N=1024,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` 调优配置文件,覆盖 batch size 1 到 4096 共 18 档,每档给出 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`;小 batch 档多用 BLOCK_SIZE_M=16,大 batch 档升至 64/128/256。

## 关键改动

```text
benchmarks/kernels/benchmark_moe.py: --batch-size 改为 nargs="+",一次跑多个 batch size 的调优
vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=NVIDIA_B200,dtype=fp8_w8a8.json: 新增 B200 FP8 E=16 tuned config(18 档)
```

CLI 参数改动:

```python
- parser.add_argument("--batch-size", type=int, required=False)
+ parser.add_argument("--batch-size", type=int, nargs="+", required=False)
```

```python
- batch_sizes = [args.batch_size]
+ batch_sizes = args.batch_size
```

## 性能声明

PR 正文未给出任何 benchmark 数字,仅声明"Confirmed successful tuning for multiple batch sizes in a single run. Config files generated correctly"。config 文件本身即调优产物(针对 `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`、B200、fp8_w8a8),PR 未注明相比默认 config 的加速比。

## 验证与回归风险

验证方式为 CLI 实测:用 `--batch-size 128 256 512` 验证多 batch size 输入,确认调优流程与 config 保存正确。风险面很小:benchmark 脚本参数解析变化只影响 `benchmarks/kernels/benchmark_moe.py` 的调用方式;新增 JSON config 仅在匹配 E=16、N=1024、NVIDIA B200、fp8_w8a8 的 fused MoE 形状时被选中,不影响其他形状或硬件的既有配置。

## 关联

无(pack 素材中未给出同区域 PR 信息)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
