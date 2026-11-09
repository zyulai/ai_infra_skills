# RFC #45997: [RFC]: Kernel-agnostic constant-stride (layer-major) KV cache layout for KV connectors

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45997](https://github.com/vllm-project/vllm/issues/45997) |
| 作者 | xiaguan |
| 创建日期 | 2026-06-18 |
| 状态 | ⚪ closed(关闭于 2026-10-19) |
| labels | stale |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |

## 一句话结论
提议为 KV connectors 新增 layer-major 常数步长布局:同一 buffer 内每层持连续切片,MLA 等 kernel 零改动即可用跨层 KV 优化。

## 提案动机
#27743 加入 "cross-layer KV blocks",使 KV connector 可一次为一个 block 暂存/传输所有层的 KV(全部层分配进一个 tensor,`use_uniform_kv_cache` / `allocate_uniform_kv_caches`)。但 MLA 后端唯一支持的物理布局是 block-major `(num_blocks, num_layers, ...)`,每层视图是跨步(非连续)切片,而 MLA kernel 要求每层连续视图——于是 #37090 对所有 MLA 后端禁用了 cross-layer KV(`get_kv_cache_stride_order(include_num_layers_dimension=True)` 返回 identity permutation 表示不支持)。结果是:今天主导分离式 prefill/decode 与 KV offloading 的模型恰好用不上该优化——DeepSeek V3/V3.2、GLM、Kimi、所有 MLA。禁用后 vLLM 回退为每层一个独立 `torch.zeros`,connector 看到的是 L 个基地址互不相干的分配,带来两个具体代价:1) GPU↔staging 层↔page 的转置需要每 block L 次独立拷贝,没有常数跨层步长可利用;2) RDMA connector 必须注册 L 个内存区域(每层一个)而非 1 个。

## 提出的方案
- 新增第二种 uniform 布局:**layer-major 连续切片**。为同 spec 层分配一个 buffer,每层获得常数步长 `S = num_blocks * page_size_bytes` 的连续切片;每层视图连续且与独立张量位等同(bit-identical),attention kernel(含 MLA)不受影响,无需 `get_kv_cache_stride_order` 支持——与既有 block-major 路径不同,这是 kernel 无关的。
- 收益机制 1:`cudaMemcpy2D` 转置(0 SM)——一次 copy-engine 2D 操作把 block 跨所有层聚合成连续 page(`dst=page_b, dpitch=blk, src=base+b*blk, spitch=S, width=blk, height=L`),每 block 一次 launch;今天是每 block L 次 memcpy。
- 收益机制 2:单一内存注册——一次分配使 connector 为所有层注册一个 RDMA 内存区域(一个 rkey)/一个 CUDA IPC handle,而非 L 个。
- 实现要点:`_allocate_kv_cache_tensors` 用一个 `torch.zeros(total)` 按 page-size bucket 切片,替代 N 个独立 `torch.zeros`;下游 `_reshape_kv_cache_tensors` 不变——连续 int8 切片支持 `.view(dtype)`/`as_strided` 且 `storage_offset` 被保留(vLLM core 已容忍非零 kv-cache storage_offset:Mamba 状态张量与 Triton fp8 scales 今天就用带 offset 的 `as_strided`)。
- 门控:新增一个弱于 `prefer_cross_layer_blocks`(该 preference 表示 block-major 布局)的 connector preference——只需要常数跨层步长、不需要 per-block 连续性的 connector 可选择加入。
- 开放 API 问题:如何通告/选择该布局——新 connector preference,还是独立于现有 block-major `stride_order[0] != 0` 信号的 backend 级 capability flag?如何把跨层步长告知 connector——显式(注册字段/回调)还是 connector 自行检测每层 `data_ptr` 的等差性?
- 纯增量、opt-in;默认分配不变。

## 优化点与收益
正文未给出吞吐/时延类量化数字。定量收益照抄:跨层聚块从每 block L 次独立拷贝变为每 block 一次 `cudaMemcpy2D`(跑在 copy engine 上,0 SM);RDMA/CUDA IPC 注册从 L 个区域变为 1 个;作者维护的 PegaFlow(跨节点 RDMA + host/SSD 分层存储的分离式 KV connector)把 CPU 侧 KV 从 layer-first 改为 page-first 后,per-block metadata 减为原来的 `1/num_layers`(每 block 一个 RDMA 描述符/一个 SSD iovec/一个 storage slot,而非每层一个)——vLLM 每层 GPU 布局到该 page 的转置正是本 RFC 使其变便宜的操作。

## 现状与落地
已关闭(关闭于 2026-10-19),labels 仅 stale。Feedback Period 为 1-2 周;作者表示乐意贡献 vLLM 侧分配改动与镜像 `tests/v1/kv_connector/unit/test_kv_cache_layout.py` 的单元测试;正文未列实现 PR,是否落地需查关联 PR。

## 关联
- 前置(cross-layer KV blocks): https://github.com/vllm-project/vllm/issues/27743
- 前置(cross-layer 对 MLA 禁用): https://github.com/vllm-project/vllm/issues/37090
- PegaFlow: https://github.com/novitalabs/pegaflow
- 参考测试:`tests/v1/kv_connector/unit/test_kv_cache_layout.py`

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
