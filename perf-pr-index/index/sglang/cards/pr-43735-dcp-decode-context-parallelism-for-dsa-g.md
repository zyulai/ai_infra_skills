# PR #43735: [DCP] Decode context parallelism for DSA (GLM-5.2 / GLM-5.3) with MTP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kpham-sgl |
| 合并 | 2026-10-13 |
| 分类 | Speculative Decoding |
| Label | documentation, amd, deepseek, hicache, run-ci, jit-kernel, run-ci-extra, release-highlight, memory-pool, bypass-fail-fast, parallel-stages |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含大量文件,另有 11 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/43735 |

## 一句话结论

让 `--dcp-size > 1` 首次支持 RoPE DSA 模型(GLM-5.2/5.3、DeepSeek-V3.2),并打通 DCP 与 EAGLE MTP 叠加,使长上下文可跨 rank 切分注意力上下文。

## 动机

此前 `--dcp-size > 1` 在 RoPE DSA 模型上完全不可用,存在三个阻断:sparse 后端不返回 LSE(无法做跨 rank 合并);index-K cache 与 MTP draft pool 按 per-rank 尺寸分配,而 allocator 发出的是加宽(dcp_size 倍)的 slot,寻址错位;TRT-LLM buffer 按单 rank 的 head 数分配。本 PR 基于 #42536 的方案(与 @vincentzed 合著)及 Flash 栈 #40433 / #40434。

## 优化点

核心设计:target KV 仍按位置 striped(保持 main 布局),但 **index-K cache 改为复制(replicated)**,由加宽的 allocator slot 寻址——每个 rank 对整个序列打分并选出全局 top-k。注意力侧新增一个 Triton 内核 `transform_index_page_table_dcp`,把全局 top-k slot 中本 rank 拥有的部分映射为本地行并 compact(`slots % DCP_SIZE == DCP_RANK`,slot 除以 DCP_SIZE 得本地 id);TRT-LLM sparse kernel 在 DCP 组 all-gather 后的 query heads 上运行并 `return_lse=True`,再由现有 MLA DCP merge 合并部分结果。decode、target verify、extend 全走此路径,无 prefix-KV gather。共享 top-k 层复用上层的 localized page table(`DcpPageTableCache`,57/78 层复用),把每步 remap 从 78 次降到 21 次。MTP draft(nextn)pool 在加宽 slot 上复制(`dcp_replicated`,kernel page 保持 64),draft 不经 DCP 组直接 attend(`is_nextn`/`is_draft_worker`),使 MTP index 共享直接喂给 draft。记账侧:index keys 与 draft pool 在所有平台上按物理 token 收 dcp_size 倍费用(删掉 NPU-only guard)。启动校验:generic DCP hook 拒绝 `--dcp-size > 1` 搭配 prefill CP;DSA override 拒绝 HiSparse、HiCache、PD disaggregation、tree drafting,并把 DSA 后端默认设为 `trtllm`。LSE combine 修复 all-empty padding 行除零(weight_sum clamp 到 >=1.0)。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa/dsa_dcp.py: 新增 localize_dcp_page_table(top-k 表复用)+ dcp_trtllm_sparse_attention(gather heads + return_lse,按 workspace 分块)
python/sglang/kernels/ops/attention/dsa/transform_index.py: 新增 transform_index_page_table_dcp Triton 内核(owned slot 筛选 + compact)
python/sglang/srt/layers/attention/dsa_backend.py: dcp_size/rank 初始化、workspace 按 dcp_size 扩容、_forward_trtllm 接 DCP 路径
python/sglang/srt/mem_cache/memory_pool.py: DSATokenToKVPool/MLATokenToKVPool 支持 dcp_replicated,index_page_size/index_buf_size 按复制语义计算
python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py: _dsa_dcp_overrides 拒绝不支持组合并默认 trtllm 后端
python/sglang/srt/model_executor/pool_configurator.py: index/draft pool 记账对所有平台乘 dcp_size(移除 _is_npu 条件)
```

## 性能声明

PR 未给出独立的吞吐加速声明(定位为打通 DCP 能力)。准确率表中附带 AIME26 生成吞吐(zai-org/GLM-5.3 FP8,8xB300,TP8,MTP = EAGLE 5 steps/topk 1/6 draft tokens):TP8 1140 tok/s;TP8+DCP2 1041;**TP8+DCP2+MTP 1876(accept len 3.51)**;TP8+DCP8 838;**TP8+DCP8+MTP 1784(accept len 3.62)**。GSM8K(8-shot,1319 题):TP8 0.912 / DCP2 0.917 / DCP2+MTP 0.919 / DCP8 0.914 / DCP8+MTP 0.919;AIME26 pass@1 99.17%(TP8、DCP2)与 95.00%(DCP8),pass@4 均 100%,PR 说明 pass@1 波动为采样噪声(200k token cap、缺 \boxed{} 等非退化输出)。

## 验证与回归风险

GPU kernel test(localize kernel,merged shards 与未分片 kernel 在 64 heads 下对比)+ 注册 4-GPU GLM-5.3 TP4+DCP2+MTP GSM8K 测试;spot checks 含 `--dcp-comm-backend ag_rs`、8192-token-per-rank pool 尾页场景。风险面:DCP2+MTP 曾暴露 draft one-shot MHA prefill 的 all-gather 键控 bug(用全局 DCP flag 而非 layer dcp_size),已在 `28bb3d6074` 修复;FlashInfer 0.7.0.post1 在 B300 上 64 heads、短 KV、>=1024 rows/call 时有 kernel bug(16/32/128 heads 干净),待上报 FlashInfer;HiCache 与 PD disaggregation 被显式拒绝为 fast-follow;其他 DSA 后端需在 DCP 下返回 LSE 才能接入;DSA 后端强制默认 `trtllm`。

## 关联

前序:#42536(DCP 方案基础)、#40433 / #40434(Flash 栈)。同区域:DSA/DCP 系列与 MTP 投机解码;HiCache、PD disaggregation 为后续 fast-follow。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
