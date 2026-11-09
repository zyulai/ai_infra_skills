# PR #2150: Eagle speculative decoding part 4: Add EAGLE2 worker

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yukavio |
| 合并 | 2025-01-02 |
| 分类 | Speculative Decoding |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计);diff 有省略(3 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/2150 |

## 一句话结论

为 SGLang 增加 EAGLE 投机解码 worker,Llama-2-7b 单请求解码从 156 token/s 提到 297(开 torch.compile 316)。

## 动机

SGLang 此前没有 EAGLE 投机解码支持(官方 EAGLE 代码单 H100 上为 200 token/s,SGLang 普通解码只有 156 token/s)。该 PR 是 Eagle speculative decoding 系列第 4 部分,补上完整的 worker 实现,使 draft 模型多步建树、目标模型验证、接受回填的整条流水线在 SGLang 内可用。

## 优化点

新增 `EAGLEWorker`(继承 TpModelWorker):draft worker 与 target worker 共享 embedding 和 lm_head,draft 侧先关 CUDA graph 初始化、再由 worker 自己重新捕获;decode 时循环执行 `speculative_num_steps` 次 draft decode——每步对上一步概率取 topk,与累计 score 相乘再选 topk 组合,逐步扩展 (b, topk, topk) 的分数/令牌/父节点列表;之后用 cutex 源码级 CUDA kernel `build_tree` 按 parent_list 构造 tree mask、positions 与 retrive_index(深度上限 10,topk 上限 10);把 draft 树作为一次 target verify 前向(greedy argmax 验证,`eagle_verify_retrive` Triton kernel 选最长接受链并回填 accept_index/accept_length),再由 `prepare_extend_after_decode` + `create_extend_spec_info` 把接受的 token 落入 KV pool 并准备下一轮 draft 输入;未接受的 draft token 的 KV slot 被释放。配套 Triton kernel 处理 req_to_token 映射(assign_req_to_token_pool)与 draft decode 的 kv indices 生成(generate_draft_decode_kv_indices)。

## 关键改动

```text
python/sglang/srt/speculative/eagle_worker.py: EAGLEWorker,draft 循环/verify/extend-after-decode 编排
python/sglang/srt/speculative/eagle_utils.py: EAGLEDraftInput/EagleVerifyInput 与 4 个 Triton kernel
python/sglang/srt/speculative/build_eagle_tree.py: cutex build_tree CUDA kernel(tree mask/positions/retrive_index)
examples/runtime/engine/EAGLE_offline_batch_inference.py: 离线批量推理示例
```

## 性能声明

单卡 H100、meta-llama/Llama-2-7b-chat-hf,单请求(temperature=0,max_new_tokens=256,PR 内附 benchmark 脚本):官方 EAGLE 代码 200 token/s;SGLang 普通解码 156 token/s;SGLang EAGLE(--speculative-algo EAGLE,--speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B,--speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 64,--mem-fraction 0.7)297 token/s;再加 --enable-torch-compile --cuda-graph-max-bs 2 为 316 token/s。

## 验证与回归风险

PR 正文无准确性评测;build_eagle_tree.py 的 `__main__` 内含 kernel 输出与 Python 参考实现的 tree_mask/positions/retrive_index 逐项 assert_allclose 自检。回归面:cutex 成为新增依赖(build_tree 用其 SourceModule 编译内联 CUDA);kernel 内 depends_order 固定 10 长度、topk<=10 硬断言;本版 verify 为 greedy argmax,不支持采样验证;draft/verify 路径深度耦合 attention backend 的 custom_mask 与 kv indices 生成,后续后端需适配。

## 关联

系列第 4 部分,正文列出的子 PR:#2678、#2684、#2709;同区域为此后的 EAGLE2 优化(如 spec overlap、flashinfer backend 适配),素材未给出编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
