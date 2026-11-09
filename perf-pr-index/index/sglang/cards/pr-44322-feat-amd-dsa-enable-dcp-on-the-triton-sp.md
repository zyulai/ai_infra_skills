# PR #44322: [Feat] [AMD] dsa: Enable DCP on the triton sparse-MLA backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @EricKing626 |
| 合并 | 2026-10-28 |
| 分类 | Speculative Decoding (tier: major) |
| Label | amd, deepseek, run-ci, jit-kernel, memory-pool |
| Diff | 素材未提供 diffstat;diff 含 18 个文件,另有 5 个文件因篇幅被省略 |
| 链接 | https://github.com/sgl-project/sglang/pull/44322 |

## 一句话结论

在 ROCm triton sparse-MLA 路径启用 DSA DCP,agentic 回放 c40 每 GPU 吞吐 4.8x、c24 2.2x。

## 动机

#43735 已让 RoPE DSA 模型在 `trtllm` backend 上支持 `--dcp-size > 1`,但 ROCm 实际走的是 `triton` DSA backend,该路径此前无法开 DCP。要在 triton 路径上补齐三块能力:triton sparse-MLA kernel 需返回 LSE(Log-Sum-Exp)以合并各 rank 的部分结果;HIP FP8 KV cache 需在 DCP 下把写落到读方查找的行;还要修复 ROCm DSA MTP draft 在 DCP 下的 GPU memory access fault。

## 优化点

正确性侧:triton sparse-MLA decode 新增 `return_lse`/`lengths` 参数,输出每行 base-2 LSE 供 DCP 按权重合并各 rank 部分结果;FP8 量化 KV 写按 `slot % dcp_size == dcp_rank` 只写本 rank 拥有的槽、写在 `slot // dcp_size` 行,与 BF16 写规则一致;ROCm DSA EAGLE/NEXTN draft 改为与 target 相同的 KV 分片与寻址规则(由 `dsa_draft_shards_under_dcp` 门控在 HIP 上),消除访问越界。性能侧:triton DCP sparse-MLA 扫描按 owned top-k 计数截断,不再全量扫 top-k 再 mask 掉约 (W-1)/W;DSA prefill indexer 的行在 DCP 组间均分、各 rank 只算 1/W 的 top-k 再 all-gather;DCP 组间传输的 Q 以 FP8 KV dtype 打包发送,字节数减半且结果不变(kernel 收到后本就会做同样转换)。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py: decode kernel 增加 LSE 输出与 per-row owned-count 截断
python/sglang/kernels/ops/kvcache/mla_buffer.py: FP8 KV 写按 DCP rank/ world size 分片落行
python/sglang/srt/layers/attention/dsa/dsa_backend.py: HIP+DCP 时 decode/verify 走 _forward_triton_dcp 本地化路径
python/sglang/srt/layers/attention/dsa/dsa_indexer.py: prefill indexer 行按 DCP 均分并 in-place all-gather top-k
python/sglang/srt/layers/dcp/comm.py: all_gather_q_for_mla_decode 支持按 FP8 dtype 打包 Q 传输
```

```python
if TOPK_TAIL_INVALID:
    num_tiles = tl.cdiv(tl.load(topk_length_ptr + t), BLOCK_K)
else:
    if USE_TOPK_LENGTH:
        valid_topk = tl.minimum(tl.maximum(tl.load(topk_length_ptr + t), 0), topk)
    num_tiles = tl.cdiv(valid_topk, BLOCK_K)
```

## 性能声明

InferenceX agentic 回放,`amd/GLM-5.2-MXFP4`,MI355X,TP4/EP4 + DCP4 + MTP(模拟 AL 3.61),每点 1 小时;基线为 10/19/2026 commit 33dd8f6a2e,本 PR 为 10/28/2026 commit b8a0a1ac70:

- c24:TPUT per Chip 10,872 → **23,416 tok/s/GPU(2.2x)**,P90 Interactivity 14.07 → **31.32 tok/s/user(2.2x)**,chip 命中率 18.6% → 96.8%。
- c40:TPUT 5,144 → **24,740(4.8x)**,P90 7.19 → **16.79(2.3x)**,chip 命中率 95.5%。
- c16:基线工作集基本放得下,TPUT 17,396 → 17,196(0.99x),P90 53.07 → 41.19(0.78x),无收益。
- c48:DCP 池满(chip 命中 82.6%)且 DCP 拒绝 HiCache,TPUT 4,781 → 10,774(2.3x)但 P90 6.30 → 2.56(0.41x)。

KV 容量:2.83M → 8.81M tokens(每 rank 2.20M 行 x 4)。

## 验证与回归风险

正确性:MI355X 上 passkey 8K/34K/96K 均 3/3,GSM8K(400 题 few-shot)no-DCP 与 DCP=4 均 0.953,MTP accept length 3.11 → 3.47(4 prompts x 400 tokens, greedy);FP8 写 kernel 通过 4 模拟 rank 测试,triton LSE kernel 级一致。风险:不开 DCP 时参数默认旧行为、CUDA 路径未动、draft 分片仅 HIP 生效;ROCm 上 `--dcp-size > 1` 强制要求 triton prefill/decode backend,否则启动报错;DCP 与 HiCache 互斥,池满后前缀被重算(c48 P90 恶化即为代价),DCP+HiCache 由后续 #45205 承接。pack 记录 CI:Base 通过、Extra 失败、AMD ROCm 10 进行中。

## 关联

前序:#43735(trtllm backend 上的 DCP 支持,本 PR 建立其上)。后续:#45205(DCP+HiCache)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
