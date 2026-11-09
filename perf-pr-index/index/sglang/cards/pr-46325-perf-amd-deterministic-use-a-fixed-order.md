# PR #46325: [Perf] [AMD] deterministic: Use a fixed-order two-stage all-reduce for large inputs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @aarnetalman |
| 合并 | 2026-11-02 |
| 分类 | Parallelism / Communication(major) |
| Label | amd, sgl-kernel, run-ci |
| Diff | 素材未注明 diffstat(diff 片段可见 3 个文件,含 1 个新增测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/46325 |

## 一句话结论

ROCm 确定性模式大输入 all-reduce 改用固定序 two-stage 内核,MI350X 上 Qwen3-8B TP8 确定性吞吐 +29–57%。

## 动机

ROCm 确定性模式下所有 custom all-reduce 都跑 1-stage 内核:每张卡要读所有 peer 的完整输入并按 rank 0..n-1 求和。这个固定顺序保证了结果与输入大小无关,但每卡读 n 倍数据——MI350X 上 Qwen3-8B 确定性 decode/prefill 在 TP4 有 48%、TP8 有 63%(8k-token 输入)的 GPU 时间耗在 all-reduce 上,TP8 确定性吞吐只有普通模式的一半左右。2-stage 内核(reduce-scatter + all-gather)读取量小得多,但默认每个 owner 从自己的 rank 起始求和,元素结果依赖于所属 rank,而所属 rank 随输入大小变化,不能直接保证确定性。

## 优化点

给 `cross_device_reduce_2stage` 增加模板参数 `kFixedOrder`(默认 false,普通模式不变):开启后 stage 1 按固定顺序 0..n-1 读取各 rank 求和(与 1-stage 内核相同),stage 2(all-gather)不变,从而在任意输入大小下与 1-stage 内核逐 bit 一致。`deterministic_all_reduce.hip` 把每个 dtype 三份 launcher 合并为一个模板 launcher:走 `deterministic_all_reduce_unreg`(输入拷入注册 buffer,含 CUDA graph 与大输入分片)且在 4 rank >= 512 KiB、8 rank >= 256 KiB(与 `CustomAllreduce::allreduce` 阈值一致)时用 32 blocks 的 fixed-order 2-stage 内核;小输入、2/6 rank 与 `deterministic_all_reduce_reg` 保持 16 blocks 的 1-stage 内核(reg 输入可能超过 IPC buffer)。

## 关键改动

```text
python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce_hip.cuh: cross_device_reduce_2stage 增加 kFixedOrder 模板参数,stage 1 固定按 rank 0..n-1 求和
python/sglang/kernels/aot/csrc/allreduce/deterministic_all_reduce.hip: 合并为模板 launcher;按 world_size/字节数选择 1-stage 或 fixed-order 2-stage
test/registered/kernels/ops/communication/test_amd_deterministic_allreduce_fixed_order.py: 新增;4/8 rank、bf16/fp16、eager 与 CUDA graph 下与 torch 固定序求和逐 bit 对比
```

```cpp
template <typename T, int ngpus, bool kFixedOrder = false>
__global__ void cross_device_reduce_2stage(...) {
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[kFixedOrder ? i : target];
    ...
```

## 性能声明

硬件 MI350X,ROCm 7.2.4(`rocm/sgl-dev:v0.5.22-rocm724-mi35x-20261021`)。

- 内核单次 all-reduce 时间(bf16,hidden 4096,50 次 CUDA graph 调用取中位):8 rank 下 1 MiB 23.9→15.4 us(1.55x)、4 MiB 77.8→31.4 us(2.47x)、8 MiB 149.2→50.7 us(2.94x)、16 MiB 291.5→88.1 us(3.31x);4 rank 下 1.37x–1.94x。阈值以下(decode 尺寸)保留 1-stage。
- E2E 确定性模式输出吞吐(request-rate inf,随机数据集):Qwen3-8B TP4 +16.6%/+20.5%/+24.4%(1k/8k/32k),TP8 +32.8%/+46.9%/+56.7%,长上下文 64k/120k +46.1%/+43.9%;Qwen3.8-27B TP8 +28.4%/+45.9%/+54.3%;TP2 不变(仍用 1-stage)。摘要口径:TP4 +16–24%,TP8 +29–57%(Qwen3-8B),Qwen3.8-27B TP8 +28–54%。TP4/TP8 TTFT 下降 19–47%。普通模式不变。
- 精度:`test_deterministic`(single/prefix/prefix-logprob/radix_cache,50 trials)与 bitwise batch-size 检查(1,275/1,275)全部通过;GSM8K 不变(Qwen3-8B TP8 93.48%,Qwen3.8-27B TP4 96.21%/TP8 96.13%)。

## 验证与回归风险

新增多进程测试注册进 `stage-c-test-large-8-gpu-amd-mi35x` 套件:在 4/8 rank、bf16/fp16、阈值两侧尺寸、eager 与 CUDA graph 下,断言 all-reduce 输出与 torch 按 rank 0..n-1(float32 累加)求和逐 bit 相等,且测试数据特意选成 float32 求和对 rank 顺序敏感。PR 还给出对照:现有(无序)2-stage 内核在 4/8 rank 的 42–44 个尺寸中有 42–44 个与 1-stage 不一致。风险面:仅确定性模式与 AMD HIP 路径受影响,kFixedOrder 默认关闭保证普通模式位级不变;2-stage 阈值和 32 blocks 参数仅在 MI350X 上测量;`deterministic_all_reduce_reg` 输入可超 IPC buffer,故强制 1-stage。pack 内 CI 状态显示 Extra 与 AMD ROCm 10 为失败态。

## 关联

前序:PR #34543(已合并,使超出 buffer 的确定性输入经 `deterministic_all_reduce_unreg` 分片处理,从而也吃到更快内核;本 PR 测量两列均包含它)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
