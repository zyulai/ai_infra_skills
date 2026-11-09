# PR #40825: [Feat] [NPU] kimi-k3: Support DCP with shared compact communication

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @qybnb |
| 合并 | 2026-11-03 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, deepseek, blackwell, npu, run-ci, jit-kernel, memory-pool, bypass-fail-fast |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);10 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/40825 |

## 一句话结论

Ascend NPU 上为 Kimi-K3(含 DSPARK 投机解码)启用 decode context parallelism:MLA KV cache 按 page 分片,DCP=2 时逻辑 KV 容量提升 1.81 倍。

## 动机

Kimi-K3 在 Ascend NPU 上需要 DCP 把 target 模型的 MLA KV cache 分片到各 DCP rank,降低给定逻辑上下文容量下的 per-rank KV 存储,并覆盖 DSPARK 投机解码路径。此前 NPU 的 DCP 校验只放行 DeepSeek DSA 模型,其余架构直接报错。

## 优化点

(1) KV 布局:target MLA latent 与 RoPE buffer 按物理 page 在 DCP rank 间交错分布,cache 写位置本地化到 owner rank;draft KV 保持 replicated(draft worker 的 dcp_size 恒为 1)。(2) prefill/extend:新 `_load_mla_prefix_cache` 先 gather 本地分片、按 `get_dcp_lens`(page 交错)trim 后经 `all_gather_kv_cache_for_dcp` 还原完整 prefix 的 token 顺序,再做既有 MLA attention。(3) decode/target verify:复用通用 Q all-gather,各 rank 用本地 KV page 经 FIA(`_forward_mla_fia`,BSND 布局、head 补到 2 的幂、`dcp_mtp_attn_mask` 按 `get_dcp_chain_spec_lens` 生成)算 partial attention,再以 compact AllToAll 合并(BF16/FP16 输出 + FP32 LSE);eager verify 无真实 query 的 idle DP 参与者返回零输出与 -inf LSE 以完成 collective。(4) 图执行:rank-local KV 长度/block table/每 query 可见性均由 page ownership 推导,replay 用 backend 准备的 local CPU lengths(FIA CPU length mirror 必需),graph-only 请求行 pad 0。(5) P/D:支持 prefill DCP=1 → decode DCP=N 的 Ascend page 传输;hybrid MLA 用通用 layer-ID pairing helper,`send_kvcache_dcp` 直接按 `num_draft_entries` 选 local target pages 与 global replicated draft pages;DSA 保留 per-entry pool layout,KDA 走既有 state-transfer 路径。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py: dense MLA DCP metadata/图态/partial attention 与 prefix all-gather
python/sglang/srt/disaggregation/ascend/conn.py: DCP P/D 传输按 target/draft 边界选 page,hybrid MLA layer 配对
python/sglang/srt/arg_groups/model_overrides/kimi_k3.py: NPU 侧 dcp_comm_backend=a2a、不复制 q_proj
python/sglang/srt/arg_groups/model_hook.py: NPU DCP 白名单加入 Kimi-K3
```

## 性能声明

- KV 容量(Kimi-K3):DCP=1 最大逻辑 KV 379,520 tokens/DP;DCP=2 达 687,872(1.81 倍,+81.25%)。
- 输出吞吐(NPU 峰值内存限制 <55109M):DCP=1 144 tok/s → DCP=4 237 tok/s(1.65 倍)。工作负载:batch size 4(另一组 bs=8、99.9% cache),128K input/1K output,radix cache disabled,Prefill-Decode colocated。
- 精度:GPQA 0.9445(截图)。
- 作者明确注明:以上模型结果为参考测量,"have not been reproduced on this head"(未在本 PR head 复现);测试条件细节以截图为主,文字未完整给出。

## 验证与回归风险

17 个 CPU 回归测试(isolated CANN 9.2 overlay)通过:覆盖 20 种 hybrid PP/draft/buffer-width 传输组合(拒绝 layout-helper 误用)、DSA index-buffer 传输地址、非法 metadata/draft-count 拒绝;CPU prefix 重建检查 450 配置、2,104 rank 输出(DCP 2/4/8、token/page 交错、partial pages、mixed/empty prefix、128K/512K 上下文)。作者明确:CPU 检查不验证原生 NPU kernel 与全模型 serving,全模型精度/吞吐/P-D 端到端在本 head 仍 pending。CI:两轮 Base 与 AMD ROCm 通过,Extra 失败(x)。

## 关联

同区域(卡片库中):#14194(DeepSeek V2 DCP,NPU DCP 前作)、#32541(Kimi-K3 模型支持)、#38943(DCP breakable prefill CUDA graph)、#47180(AMD DSA DCP query 复制);DSPARK 基础见 #33561。
