# PR #35629: [NPU] Adapt DFlash2 speculative decoding to Ascend NPUs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @syd520zy |
| 合并 | 2026-09-07 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, speculative-decoding, npu, run-ci, jit-kernel, bypass-fastfail |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 cookbook、dflash_worker_v2 与 2 个单测文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/35629 |

## 一句话结论

把 DFlash2 投机解码适配到 Ascend NPU:修复 verify 元数据、verify 走 argmax,Qwen3.8-27B TP2 上 e2e 加速 1.75x-3.05x。

## 动机

DFlash2(#35371,block-diffusion drafter + candidate selector)的 checkpoint 此前无法在 NPU 上服务,因为 target verification 路径假设了 CUDA 行为;第一代 DFlash 已在 NPU 上运行(#23122、#31739)。本 PR 适配 DFlash2 的 draft 与 verify 路径,同时保持 CUDA 行为不变。运行环境依赖 `triton-ascend`(假定 runtime 已带 Triton)。

## 优化点

三处 NPU 专属决策。其一,target verify 元数据:非 NPU 保持 `skip_attn_backend_init=True`(prepare_for_verify 已规划 attention 元数据);NPU 传 `None` 让 target forward 初始化,避免 NPU graph runner 读取未初始化的 `raw_num_token`——因为 NPU 的规划必须推迟到 DP/EP padding 定型之后。其二,selector 采样验证:NPU 上跳过全词表 selector sampling verify,改用 argmax;这不是 selector 算法限制,而是性能策略——全词表路径要 scatter 稠密 `[batch_size, gamma, vocab_size]` 概率缓冲并做词表级运算,实测增加约 6 ms TPOT(51 ms vs 45 ms),对无损 greedy 请求没有必要。其三,draft 与 verify 一致性:selector sampling 能力在 worker 初始化时解析一次(`_selector_sampling_enabled = not _is_npu`),graph 捕获与 eager 路径消费该状态而非重复查设备;NPU 上强制两条 draft 路径 greedy proposal,避免随机 draft 被 greedy target 验证而缩短接受长度。非 greedy 请求被接受但一次性 warning 后整体回退 greedy(采样分布不保留);DSpark NPU 采样支持不在本 PR 范围。

## 关键改动

```text
python/sglang/srt/speculative/dflash_worker_v2.py: _selector_sampling_enabled 能力位、_SelectorDraftSampler 采样开关、verify 的 skip_attn_backend_init NPU 分支
docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx: DFlash2 NPU 行为与 greedy-only 无损边界文档
test/registered/unit/spec/test_dflash_logits.py: sampling fallback warning / greedy 强制 / accept 回退用例
test/registered/unit/spec/test_dflash_extra_buffer_lazy.py: 隔离 generic/CUDA hook 顺序测试与 NPU DSV4 bundle
```

```python
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
-            skip_attn_backend_init=True,
+            skip_attn_backend_init=True if not _is_npu else None,
        )
```

## 性能声明

Ascend 910C,TP2,目标 `Qwen/Qwen3.8-27B`(BF16)+ draft `z-lab/Qwen3.8-27B-DFlash2`,`--speculative-num-draft-tokens 8`,`--attention-backend ascend`,GSM8K zero-shot greedy(temperature 0.0 / top_p 1.0 / top_k 1),max_new_tokens 2048,每并发 128 样本,双方均禁 RadixCache。输出吞吐(tok/s)与加速比:

| 并发 | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 33.182 | 60.925 | 109.544 | 176.193 | 247.068 |
| DFlash2 | 101.249 | 162.244 | 259.899 | 329.448 | 431.172 |
| 加速 | 3.05x | 2.66x | 2.37x | 1.87x | 1.75x |

加速比随并发从 3.05x 降到 1.75x(高并发下 target batching 与投机开销占比上升)。profile:selector top-k 约 0.04 ms/step;target verify 受 TP2 每层 all-reduce 下限约束(每步约 43 次集合通信);全词表 selector sampling verify 约增加 6 ms TPOT。

准确性:GSM8K 精度 baseline 与 DFlash2 在并发 1/2/4/8/16 下完全一致(0.820/0.836/0.836/0.828/0.820);窗口平均接受长度约 5.8(块大小 8),接受率约 0.68。

## 验证与回归风险

运行时验证:greedy 三次重复输出 hash 一致;NPU graph 与 eager 两条路径均验证;非 greedy 请求完成并触发一次性 warning;确定性非贪婪验证未做(测试镜像缺 `batch_invariant_ops`)。单测覆盖 sampling 禁用时的 warning、能力位传播、greedy 强制、accept 回退等。风险:NPU 上仅保证 greedy(temperature=0 / top_k=1)无损;`skip_attn_backend_init` 在 NPU 语义变化依赖 verify 路径自行初始化元数据;review 中明确保留 NPU guard 而不复用 `is_dflash_sampling_verify_available()`(以免改变 HIP 等后端既有行为)。CI:Base 通过,Extra 与 AMD ROCm 7.2 失败(素材所示)。

## 关联

前序:#35371(DFlash2 上 main)、#23122、#31739(第一代 DFlash 的 NPU 支持)、#16818(精度测试方法参照)。同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
