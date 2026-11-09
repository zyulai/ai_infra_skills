# PR #11133: [Feature] Qwen3-Next & FLA: Support MTP topk>1; Up to 6% faster

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @byjiang1996(co-authored-by @hebiao064) |
| 合并 | 2025-11-02 |
| 分类 | Speculative Decoding (tier: major) |
| Label | high priority, run-ci |
| Diff | pack 未提供 diffstat(diff 因超长截断,2 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/11133 |

## 一句话结论

为 Qwen3-Next/FLA 线性注意力打通 MTP eagle-tree topk>1 草稿,acc_length 3.35 提升到 4.41。

## 动机

Qwen3-Next 与 FLA(gated delta rule 线性注意力)路径上,MTP 投机解码此前只支持 topk=1:草稿 token 只能串行 conv/attn 到 N-1、N-2 等前序 token,无法利用 eagle tree 分支结构,accept length 受限。本 PR 让树状草稿的每个 token 按父子关系 attend/conv 到正确的祖先,从而支持 topk>1 拉高接受长度。

## 优化点

核心机制:为每个 draft token 建父指针列表(`retrieve_next_token`/`retrieve_next_sibling`,并在 `_causal_conv1d_update_kernel` 内融合算出 `retrieve_parent_token`),递推状态改为从父 token 的 state 计算 `S_current = S_parent + V_current·K_current^T`。两处 Triton kernel 各加 `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK` heuristic:fused recurrent kernel 在每步从 `intermediate_states_buffer` 按 `retrieve_parent_token` 加载父步 state;causal conv1d update kernel 沿父链回溯 KERNEL_WIDTH 个祖先做卷积(链内不足时回退读 conv state 列)。其次,eagle verify 后 mamba state 回写从"按 `accept_length-1` 步覆盖"改为按每请求最后 accepted token 的 index 覆盖(topk>1 时 accepted token id 可能是 [0,2,4,6,7],取 7 而非 4)。另移除两个已不需要的 `tl.debug_barrier()`(Triton compiler bug 已修复),conv1d 输出改用 `torch.empty_like(x)` 不再原地覆盖输入。

## 关键改动

```text
python/sglang/srt/layers/attention/fla/fused_recurrent.py: recurrent kernel 支持树状父指针,按父步加载 state
python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py: conv update kernel 融合计算并沿父链卷积
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: metadata 携带 retrieve 三元组;cuda graph capture/replay 适配;topk>1 时按 accepted_indices 回写 mamba state
python/sglang/srt/speculative/eagle_worker.py: verify 后向 hybrid_linear_attn_backend 传每请求最后 accepted index
python/sglang/srt/layers/attention/mamba/mamba2_metadata.py: ForwardMetadata 增加 retrieve 三字段
```

```python
if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
    # when calculating current step's attention, load the state from the parent token
    parent_step_idx = tl.sum(
        tl.where(token_indices == step_idx, parent_idx_tokens, 0)
    )
    step_offset = parent_step_idx * HV * K * V
    cache_ptr = (
        intermediate_states_buffer
        + cache_idx * cache_steps * HV * K * V
        + step_offset
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)
```

## 性能声明

测试条件:H100 x 4,Qwen3-Next-80B-A3B-Instruct,`--speculative-algo NEXTN --attention-backend fa3`,`python3 -m sglang.test.send_one`。

- Before(topk=1,num_steps=3,draft=4):acc_length=3.35,speed=343.33 tok/s
- After(topk=3,num_steps=5,draft=8):acc_length=4.41,speed=349.29 tok/s,PR 正文写"Up to 1.7% faster"(标题中的 6% 与正文数字不一致,以正文为准)

Accuracy(gsm8k `bench_sglang.py`,TP4 MTP):835 禁用 cuda graph 0.965;835 cuda graph 0.955;835 DP2 cuda graph 0.950;314 cuda graph 0.950。

## 验证与回归风险

PR 依赖 gsm8k 精度回归与 send_one 单请求基准;未见新增单元测试(pack diff 未见 test 文件)。回归面:Qwen3-Next/FLA hybrid linear attention 的 decode/verify 路径、MTP/EAGLE 投机解码与 cuda graph capture/replay 组合;conv1d 输出从原地写改为 `empty_like` 影响所有走 `_causal_conv1d_update_kernel` 的调用方;移除 debug_barrier 依赖新版 Triton 编译器行为。

## 关联

pack 正文未列出关联 PR 编号。同区域:Qwen3-Next/FLA MTP 投机解码、hybrid_gdn_config 线性注意力后端。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
