# PR #9873: [Feature] Speculative decoding support lookahead

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @a4zhangfei |
| 合并 | 2025-09-18 |
| 分类 | Speculative Decoding(tier: major) |
| Label | enhancement, high priority, speculative-decoding, run-ci |
| Diff | 素材未提供 diffstat(20 个文件因体积原因从 diff 中省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/9873 |

## 一句话结论

引入 LOOKAHEAD 投机解码:用 C++ Trie 复用输出局部性生成 draft 树,无需 draft 模型,高局部性小模型最高 2.61× 加速。

## 动机

对输出局部性好的 LLM(如意图识别、行为路由等短输出、高重复场景),lookahead 式投机解码能以较低代价获得显著加速。此前 sgl-project/sglang#2790 已讨论该方向,本 PR 参考 #2790 将 LOOKAHEAD 作为 `--speculative-algorithm` 的新选项落地。

## 优化点

新增 LOOKAHEAD 投机解码栈:核心是 C++ 实现的 token Trie(`cpp_lookahead/lookahead.cpp`),节点带全局 LRU 与按频率排序的 children,后台线程异步插入生成序列,容量满时按 LRU `squeeze`;匹配时对最近 token 串按 min–max match window 滑窗查 Trie,支持 BFS(按窗口深度分配 bfs breadth)与 PROB(按子节点频率归一化概率)两种建树策略,输出 draft token 序列与树形 attention mask。调度侧新增 `LOOKAHEADWorker`(无 draft 模型、无 draft KV pool),`spec_info` 类型扩展 `LookaheadVerifyInput`,FlashInfer backend、CUDA graph 捕获/回放与 `can_run` 均适配。约束:仅 CUDA、禁 overlap schedule 与 mixed chunk、不支持 DP attention,`eagle_topk>1` 且 `page_size>1` 仅允许 flashinfer backend。

## 关键改动

```text
python/sglang/srt/speculative/cpp_lookahead/lookahead.cpp/.h: C++ Trie 与 BFS/PROB 建树、树 mask 生成(新增)
python/sglang/srt/speculative/lookahead_utils.py、lookahead_worker.py: LookaheadVerifyInput 与 LOOKAHEADWorker(新增,20 个文件被省略,依据正文与引用推断存在)
python/sglang/srt/server_args.py: LOOKAHEAD 校验与 7 个 --speculative-lookahead-* 参数
python/sglang/srt/managers/scheduler.py、schedule_batch.py、model_executor/*: 挂接 draft worker、spec_info 类型与 memory pool 适配
python/sglang/srt/layers/attention/flashinfer_backend.py: spec_info Union 类型扩展
```

```python
if self.speculative_algorithm == "LOOKAHEAD":
    ...
    self.disable_overlap_schedule = True
    self.enable_mixed_chunk = False
    self.speculative_eagle_topk = self.speculative_lookahead_max_bfs_breadth
    if self.speculative_num_draft_tokens is None:
        self.speculative_num_draft_tokens = (
            self.speculative_lookahead_max_match_window_size
        )
```

## 性能声明

自研小模型(短输出高局部性场景),num prompt 256,最优配置平均 accepted length 约 4.8:concurrency 1:413.14→1079.52(speedup 2.61)、2:672.43→1489.14(2.21)、3:859.09→150.29(1.92,原表数值如此)、4:996.53→1799.86(1.81)。公开测试集:Qwen2.5-Coder-7B-Instruct + qwen2.5_test_python 数据集(1024 prompts,需转 sharedgpt 格式),H20 tp2,`--speculative-num-draft-tokens 16`,lookahead 参数 min/max match window 1/16、min/max bfs breadth 1/10、branch length 18:conc 1:265→429(accept 1.45,speedup 1.62)、2:507→782(1.47,1.54)、3:746→987(1.43,1.32)、4:971→1126(1.43,1.16)。表格未注明指标单位。另:qlen 型 mask 相比 full 型树 mask 有 8% 性能提升,但需 FlashInfer 支持(正文以提问形式给出)。

## 验证与回归风险

Checklist 勾选了单元测试项,但可见 diff 中无测试文件(20 个文件被省略);精度方面仅声明该功能已在多个业务场景运行两个月余,稳定性与加速效果已验证,无标准 accuracy 数字。回归风险:改动触及公共路径——`spec_info` 类型 Union、`prepare_for_decode`、CUDA graph `can_run`、memory pool 初始化条件(改为仅 eagle/standalone)、`flush_cache` 改经 `draft_worker.clear_cache_pool()`、tokenizer_manager 输入长度校验预留 draft token 数、bench_serving bos_token 判空;EAGLE/STANDALONE 路径行为应不变,但 standalone+DP attention 由 assert 改为 raise。LOOKAHEAD 自身禁用 overlap schedule 与 mixed chunk,吞吐基线因此低于普通 decode。

## 关联

前序:#2790(lookahead 方案讨论来源)。同区域:既有 EAGLE/EAGLE3/NEXTN/STANDALONE 投机解码栈(本 PR 并列扩展)。后续 PR 未在素材中提及。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
