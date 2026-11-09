# PR #42557: [Feat] paged-experts: Serve MoE models whose weights do not fit VRAM

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @N3mezis |
| 合并 | 2026-10-19 |
| 分类 | MoE (tier: major) |
| Label | documentation, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用(另有 11 个文件因超长未包含在 pack 中) |
| 链接 | https://github.com/sgl-project/sglang/pull/42557 |

## 一句话结论

MoE 专家权重分页:GPU 保留 K 个常驻专家、其余按需从 pinned host RAM 换入,让权重超 VRAM 的 MoE 模型可服务,单流 decode 较 `--cpu-offload-gb` 快 12–30×。

## 动机

笔记本/游戏本常见 4–16GB VRAM 配 32–64GB RAM,而 MoE 已是开源权重默认架构(Qwen3/3.5 MoE、DeepSeek、GLM、gpt-oss、Mixtral),专家权重占参数大头——Qwen3-30B-A3B int4 仅专家就约 15GB,占满 16GB 卡。现状要么加载失败,要么用 `--cpu-offload-gb` 把每个 offload 权重每次 forward 全量流过 PCIe。llama.cpp/Ollama 的做法是把部分 MoE 层专家留在系统 RAM 并在 CPU 上计算(`--n-cpu-moe`)。本 PR 走第三条路:每个 token 只路由到 `top_k` 个专家且相邻 token 大量复用,故在 GPU 上维护 K 槽专家表、host 存全部 E 个、只换入缺失的——HiCache 思路套到专家权重上。K 同时是显存旋钮:与 KV cache 直接互换(更小 K 换更长上下文/并发,更大 K 换速度)。本 PR 是 #29971(已关闭,约 7k 行)裁剪出的可评审第一步,只做核心:换页、GPU 计算、结果不变。

## 优化点

1. **K 槽 GPU 常驻 + LRU**:每层在 GPU 保留 K 个专家槽(`--paged-experts-num-resident`,或由 checkpoint 字节数 + `--mem-fraction-static` + KV 预留自动算),LRU 决定驱逐。
2. **双执行器**:`EagerExecutor` 在 host 规划、多 wave 服务任意 step(可超出 K 个不同专家),在 breakable prefill graph 中作为 eager break;`DeviceExecutor` 用新 JIT kernel 在 GPU 上一步完成 decide + page-in,无 host sync,因此能被 decode CUDA graph 捕获(支持至 `K // top_k` 个并发请求)。
3. **新 JIT kernel**(`kernels/jit/csrc/moe/paged_experts.cuh`,约 290 行含 wrapper,不动 sgl-kernel):单 warp 的 on-device LRU decide kernel 写出 page-in 计划(src/dst/count 全在 device buffer);一个 gather launch 经 UVA 地址从 pinned host store 拷贝该层所有分页张量,count 在 device 上读取,replay 的 graph 精确搬 decide 选出的专家。
4. **结果不变机制**:runner(triton/Marlin)以 `no_combine` 返回按 router 加权的 per-(token, expert) 输出,由 RunnerContract 执行与 runner 相同的 top-k 求和——这是多 wave 步骤 bit-identical 的关键;新增 `no_combine_keep_router_weight`、`moe_combine_topk`(triton 尾部拆分)、`fused_marlin_moe(no_combine=)`。
5. **格式/加载**:每量化方法一个 `ExpertFormat`(bf16/fp16、block-FP8、GPTQ-Marlin);GPTQ 的 Marlin repack 对 host store 按 K 个专家一批运行;权重经 SGLang 原生 per-expert loader 加载进 host store(`FusedMoE.weight_load_method` 重定向)。

## 关键改动

```text
python/sglang/srt/layers/moe/paged_experts/(新包,约 1150 行): method.py
  (PagedExpertsMoEMethod 接线与启动守卫)、formats.py(每量化一个
  ExpertFormat)、runners.py(RunnerContract)、store.py(pinned host
  store)、residency.py(LRU/wave 规划 + DeviceResidency)、
  executor.py(Eager/Device 双执行器)、sizing.py(自动 K)
python/sglang/kernels/jit/csrc/moe/paged_experts.cuh + kernels/ops/moe/
  paged_experts.py: on-device LRU decide + 多张量 UVA gather kernel
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: weight_load_method
  与 paged-experts hook;_load_w13/weight_loader 按 layout method 分派
python/sglang/srt/arg_groups/fields/exec_.py: --enable-paged-experts 与
  --paged-experts-num-resident 两个新 flag
```

## 性能声明

PR 正文,未独立核实。机器:RTX 5070 Ti(16GB,PCIe 5.0 x16)与 RTX 4070 Laptop(8GB,PCIe 4.0 x8);workload 为 ShareGPT 真实 prompt、256 输出 token、两侧相同 KV 容量(10,240 tokens/4 slots)、byte-matched 量化、llama.cpp 调到最优 `--n-cpu-moe`:

- **vs SGLang `--cpu-offload-gb`**:16GB 卡单流 decode 快 **12×**(int4:205.6 vs 16.6 tok/s)与 **30×**(FP8:87.7 vs 2.9 tok/s);8GB 笔记本快 **27×**(32.6 vs 1.2 tok/s)。
- **vs 调优 llama.cpp**:16GB int4 decode 205.6–205.9 vs 194.4 tok/s,4 并发 out tok/s 374–387 vs 312;16GB FP8 decode 87.7 vs 45.2 tok/s;8GB 笔记本单流 30.6–32.9 vs 23.0–32.4 tok/s,大体接近或略优。
- **长 prompt TTFT 占优**(2,048-token):267 vs 344ms(int4 16GB)、443 vs 984ms(FP8 16GB)、1,441–1,538 vs 2,396ms(笔记本);**短 prompt TTFT llama.cpp 占优**(89–90 vs 68ms 等,prefill 需一次性换入全部缺失专家,约 4GB/Qwen3-30B int4 K=92)。
- **已知反例**:小 K/E 下多并发每步路由专家超出容量会掉速——笔记本 4 并发 11–12 vs llama.cpp 41 tok/s(K=26/128);FP8 16GB K=40 时 84 tok/s、K=24 时仅 31(llama.cpp 55)。
- **精度**:bf16 与 FP8 logprobs 对未分页模型 **bit-identical**(eager 与 CUDA graphs,K=8 多 wave);GPTQ 在 Marlin 自身 batch 噪声内(max 0.1, mean 0.01)。GSM8K:16GB int4 0.955 vs llama.cpp 0.965(IQ4_XS),FP8 0.945 vs 0.955(Q8_0),笔记本 int4 0.98 vs 0.98。

## 验证与回归风险

- 测试三层:CPU 单测(paged vs unpaged、LRU/waves、host-loader 布局、K-at-a-time repack、sizing);CUDA kernel 测试(decide 对 Python 参考实现 200 随机步、gather、20 次 CUDA-graph replay 变路由、超 K 拒绝);e2e(bf16/FP8/GPTQ,eager 与 CUDA graphs 对比 unpaged)。实测于 RTX 5070 Ti (SM120) 与 RTX 4070 (SM89),用 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 跳过版本检查,CI 镜像结果可能不同。CI 状态(pack 记录):Base/Extra/AMD ROCm 均在进行中。
- 限制(正文):host store 必须 pinnable(WSL2 上限约为系统 RAM 一半);GPTQ 启动慢(Marlin repack 按 K 批 + 首次 JIT 编译可达分钟级,可能超 scheduler watchdog);decode 图捕获上限 `K // top_k` 请求,超出会 raise。
- 回归面:triton/Marlin runner 的 no_combine 语义改动触及所有 MoE 用户(默认关闭,仅在 paged-experts 开启时启用);`FusedMoE` weight loader 分派路径变化;上限内 decode 走 DeviceExecutor 但 routing 超容量时 EagerExecutor 回退。

## 关联

#29971(前序 PR,已关闭;本 PR 为其裁剪版,遗留项均声明为后续工作)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
