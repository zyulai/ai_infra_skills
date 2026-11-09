# PR #2192: Custom all reduce kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hanzhi713 |
| 合并 | 2024-01-27 |
| 分类 | Attention / Kernel |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/2192 |

## 一句话结论

引入 vLLM 自研 custom all-reduce CUDA 内核(IPC 直连 + 单/两阶段/蝶形算法),TP 推理时延场景替代 NCCL,Llama-70B TP4 端到端时延 -11%。

## 动机

小 batch 推理时 TP all-reduce 是延迟瓶颈,NCCL 为通用训练设计、每层调用开销大且内部缓冲会被 CUDA graph 捕获放大显存占用。作者在独立 writeup(Latency-optimal allreduce and cuda graph optimization.pdf)中验证了一对多 p2p 直写同步 + 定制 reduce 内核的方案,本 PR 落地为 `CustomAllreduce` 并自动兼容 CUDA graph。

## 优化点

通过 `cudaIpcMemHandle` 把各 rank 的输入/临时 buffer 指针互相注册,内核内直接跨卡读写;跨 device 同步用 Metadata 里的 Signal 自旋标志——各 rank 同时向对端 signal 的对应字节写 255,延迟仅一次 p2p write,`end_sync` 用原子计数保证最后一个 block 读完才放行。三种算法按拓扑与尺寸分派:world_size=2 或 full NVLink 小输入(<512KB/256KB)走 one-stage 全量 reduce;NVLink 大输入走 two-stage(reduce-scatter + allgather);非 NVLink(PCIe)走 half-butterfly 分组 all-to-all + 跨组相加。数据通路用 packed 类型强制生成 `ld.128/st.128`,fp32 累积再降转,各 rank 累加顺序一致保证 bitwise 一致。块数固定 36(grid search 结论,过多 SM 会争抢 NVLink 总线)。CUDA graph 兼容:capture 时把未注册 buffer 记录,事后 `register_graph_buffers` 批量做 IPC 注册。

## 关键改动

```text
csrc/custom_all_reduce.cuh: CustomAllreduce 核心(同步原语 + 3 种 reduce 内核 + IPC/graph buffer 注册)
csrc/custom_all_reduce.cu: torch 绑定(should_custom_ar 尺寸/连续性门控,all_reduce_reg/unreg)
csrc/pybind.cpp, csrc/ops.h: custom_ar 子模块暴露
vllm/config.py: 新增 --disable-custom-all-reduce 开关
```

## 性能声明

`benchmark_latency.py --input-len 256 --output-len 256`(L=时延,M=显存):Llama-70B A100 TP4 BS64 时延 16.37 → 14.57 s,显存 76723 → 75717 MB;Llama-33B A100 TP2 BS64:15.88 → 14.38 s;Llama-13B A30(无 NVLink)TP2 BS32:10.43 → 9.79 s;Llama-7B T4 TP2 BS32:17.90 → 17.51 s。吞吐:70B TP4 3.68 → 3.87 req/s、1761.33 → 1852.65 tok/s。PR 指出:NVLink 系统收益更大,batch 越小(allreduce 延迟主导)收益越明显。

## 验证与回归风险

`csrc/custom_all_reduce_test.cu` 提供 MPI+NCCL 独立对拍(数值误差与性能对比);端到端模型正确性在 TODO 中勾选完成。已知限制:输入字节数须为 16 的倍数(自动检查)、仅 out-of-place、不支持 TP+PP 混合并行、world size 仅 2/4/6/8(奇数不支持)、仅 fp32/fp16/bf16。回归面:`CustomAllreduce.__init__` 修改了所有 TP worker 的初始化路径。

## 关联

同区域:vLLM TP 通信基石 PR;后续 QuickAllReduce(MI300)、PR #20759(PyTorch Symmetric Memory)在同一 `CudaCommunicator.all_reduce` 分派链上与其并列;尺寸上限表的演进见 #20759。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
