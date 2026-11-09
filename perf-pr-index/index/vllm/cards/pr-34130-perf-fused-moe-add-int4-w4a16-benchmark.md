# PR #34130: [Perf] fused_moe: add int4_w4a16 benchmark support and tuning config

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgehre-amd |
| 合并 | 2026-02-13 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含 benchmark_moe.py 修改与 1 个新增 tuning config JSON |
| 链接 | https://github.com/vllm-project/vllm/pull/34130 |

## 一句话结论

给 fused MoE benchmark/tuning 脚本加 int4_w4a16 量化支持,并为 AMD Strix Halo(Radeon 8060S)落地调优配置:Qwen3-30B-A3B w4a16 上 Median TTFT 快 22%、TPOT 快 5%。

## 动机

int4_w4a16(AWQ/GPTQ/compressed-tensors 的 w4a16 MoE)此前没有 benchmark/tuning 入口:脚本无法生成 uint8 打包的 int4 权重与 group-wise scale,tuning 搜索空间也不适配 gptq_awq kernel 的约束,导致该量化格式在 AMD iGPU(Strix Halo)上没有调好的 Triton MoE kernel 配置可循。

## 优化点

benchmark_moe.py 三方面:一是 benchmark_config/benchmark/tune/save_configs 全链路新增 use_int4_w4a16——权重按 2 个 int4/uint8 打包(K 维减半,w1 (E, shard_intermediate, K/2)、w2 (E, hidden, N_inter/2)),scale 为 (E, N, K//group_size) fp16,并要求 block_quant_shape;二是 get_quantization_group_size() 从 HF config 自动提取 group_size(支持 AWQ/GPTQ 顶层键与 compressed-tensors 的 config_groups.*.weights.group_size),int4_w4a16 时 block_quant_shape=[0, group_size](N 维不分块);三是 tuning 适配:int4 权重按 uint8 打包,搜索空间按非 fp16 处理(不做 matrix_instr_nonkdim/kpack 探索),gptq_awq kernel 对任意 BLOCK_SIZE_K 均成立故跳过 group_size 过滤以保留全空间(如 BLOCK_SIZE_K=64 + group_size=128),SPLIT_K 固定为 1(kernel 必需 constexpr,运行时只用 1)。--dtype 增加 int4_w4a16 选项。另附 Radeon 8060S 上 E=128,N=768(如 Qwen3-30B-A3B MoE 层)batch 1-32 的调优 JSON。

## 关键改动

```text
benchmarks/kernels/benchmark_moe.py: int4_w4a16 权重/scale 生成、group_size 自动提取、tune 搜索空间与 SPLIT_K=1 固定
vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json: 新增 batch 1-32 调优配置(triton 3.6.0)
```

## 性能声明

测试条件:AMD Strix Halo(gfx1151,LPDDR5X-8000 128GB),模型 RedHatAI/Qwen3-30B-A3B-Instruct-2507.w4a16,input-len=128、output-len=128、num-prompts=5。Median TTFT 182.17ms -> 141.69ms(22% faster);Median TPOT 41.88ms -> 39.73ms(5% faster)。对比两侧即"无调优配置 vs 使用本 PR 新增的 int4_w4a16 调优配置"(kernel 级微基准数字正文未列,仅给出 E2E 两个中位数)。

## 验证与回归风险

无新增自动化测试(改动集中在 benchmark/tuning 工具与配置文件);调优 JSON 由脚本产出。回归面:benchmark_moe.py 的 CLI(--dtype 新增枚举)与 tune 流程对既有 fp8_w8a8/int8_w8a16 路径的参数签名改动(is_fp16 判定、save_configs 签名);运行时 MoE kernel 本身未改,风险限于工具链使用者。

## 关联

同区域:triton fused MoE kernel 的其他量化 tuning(benchmark_moe.py 既有 fp8_w8a8/int8_w8a16 路径);正文未给出关联 PR 编号。
