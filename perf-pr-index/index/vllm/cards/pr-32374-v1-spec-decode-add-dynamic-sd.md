# PR #32374: [V1][Spec Decode] Add Dynamic SD

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ekagra-ranjan |
| 合并 | 2026-06-14 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, speculative-decoding, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾标注 "5 file(s) dropped from diff for size"(见 PR 完整 diff),diff 保留 19 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/32374 |

## 一句话结论

V1 新增 Dynamic SD 动态调 draft 长度 K,1×H100 高 BS 下 TPOT 优于静态 EAGLE(BS256: 19.96→17.2 ms)。

## 动机

SD 方法在 decode 时每序列要 verify K tokens,batch size 增大时有效 BS 变 BS*K,验证算力需求上升;当 BS*K 超过临界 BS,SD 反而恶化 TPOT。DSD 通过把 K 下调到最优值,使系统继续从 SD 获益。正文给出两类用例:同一部署面对高/低变化的 workload(K 随负载升高而下降),以及 RL rollout(起始高 BS、末尾因长尾请求只剩很少 BS,K 随之上升)。正文注明 V0 有 milestone 0、V1 此前没有任何 Dynamic SD,本 PR 实现介于 Dynamic SD(DSD)issue #4565 的 Milestone 2 与 3 之间。

## 优化点

用运行时信息(batch size、position level acceptance rate)结合离线 profile 的参数(cold start 用的 token acceptance rate、draft 与 target model 的相对成本)动态决定 speculative decoding 的 proposed length。推理前用代表性数据集 profile 两件事:position-level acceptance rate(解决冷启动)与 draft/target 成本;运行时用当前 batch size 与迄今观测到的平均 position-level acceptance rate(等 `warmup_steps` 后才用实测 AR,之前用离线 AR)。goodput 按 TurboSpec 思路但简化为 `goodput = AL / ITL`(AL 为 K 的函数、ITL 为 K 与 BS 的函数),直接记录 SD model(target + draft)的 ITL across batch sizes,在相邻 BS 与 K 之间线性插值,避免按模型配置 / KV cache 大小建回归模型。实现上 `dynamic/generate_config.py` 为主控:调 `offline.py` 取各位置 AL、用 `vllm bench sweep` 跑不同 BS/K 的 ITL、汇总成 `dynamic_speculative_config.json`;`DynamicSpeculativeConfig`(`config/speculative.py`)持有配置与路径;`dynamic/manager.py` 的 Dynamic SD Manager 读取 ITL、按 BS 与 K 插值生成最优 K 并在 proposal 时提供给 SD 方法。

## 关键改动

```text
vllm/v1/spec_decode/dynamic/generate_config.py: 主控脚本,四步(取 position-level AL、vllm bench sweep 跑 ITL、汇总、保存 dynamic_speculative_config.json)
vllm/v1/spec_decode/dynamic/manager.py: Dynamic SD Manager,运行时按 BS/K 插值 ITL 生成 optimal K 供 proposal 使用
vllm/v1/spec_decode/dynamic/utils.py, vllm/v1/spec_decode/offline.py: 动态配置工具与离线 AL 统计(offline.py 为 offline_inference/spec_decode.py 迁入 vllm/ 以便 CI 导入)
vllm/config/speculative.py: 新增 DynamicSpeculativeConfig 配置类(batch_stats、max_num_speculative_tokens、acceptance_rate_per_pos、config 路径)
vllm/v1/core/sched/scheduler.py, async_scheduler.py, sched/output.py: 调度侧接入动态 K(diff 文件清单);docs/features/speculative_decoding/* 新增文档
```

注:pack 末尾标注 5 个文件因超长被丢弃;上述文件作用来自 PR 正文 "File changes" 段与 diff 文件清单。

## 性能声明

Offline profiled on MTBench / Tested on MTBench(1xH100、`llama 3.1 8b`、TPOT ms,越低越好):BS 1 Vanilla 6.3 / EAGLE 3.98 / Dynamic EAGLE 3.98;BS 4 6.38 / 4.03 / 4.05;BS 16 6.77 / 4.45 / 4.45;BS 64 7.94 / 6.78 / 6.56;BS 128 10.15 / 11.19 / 9.88;BS 256 16.2 / 19.96 / 17.2。正文结论:低 BS 时 DSD 与 SD 相当且都优于 vanilla;高 BS 时 SD 差于 vanilla、DSD 优于 SD 且更接近 vanilla,但 DSD 仍有"即使 K=0 也跑 draft model prefill"的开销(为后续 BS 变化预填 token)。Offline profiled on MTBench / Tested on InstructCoder(同 1xH100):BS 128 Vanilla 12.69 / EAGLE 11.55 / Dynamic EAGLE 11.85 / Dynamic EAGLE with runtime AL 11.43;BS 256 21.19 / 21.5 / 21.07 / 21.07;正文称加 runtime AL 仅带来小幅改善,说明 MTBench 数字对 InstructCoder 可迁移。Cmds 段给出 `generate_config.py`(`--method eagle --model-dir meta-llama/Llama-3.1-8B-Instruct --draft-dir yuhuili/EAGLE-LLaMA3.1-Instruct-8B --tp 1 --batch-size-list 1 4 16 64 256 --num-speculative-tokens-list 1 3 5 --dataset-name hf --dataset-path philschmid/mt-bench`)与 vanilla / Eagle / Dynamic Eagle 的 serve 和 `vllm bench serve`(MTBench、`--num-prompts = 20 * MAX_CONCURRENCY`、`--max-num-seqs 256`)命令;示例 `dynamic_speculative_config.json` 含 `batch_stats`、`max_num_speculative_tokens: 5` 与 `acceptance_rate_per_pos`。注:pack 的 `perf claims detected in body (regex)` 为 "(none detected by regex)",但正文 Results 含上述 TPOT 表格;PR 正文未注明精度字段。

## 验证与回归风险

PR 正文给出两组 TPOT 对比表(MTBench / InstructCoder)、`generate_config.py` 与 benchmark 命令、示例 `dynamic_speculative_config.json`,并说明 `offline.py` 也用于 CI 测试;未列出 pytest 单元测试命令或精度(eval)结果。风险面:改动 `config/speculative.py`(新增配置类)、`vllm/v1/core/sched/scheduler.py` / `async_scheduler.py` / `sched/output.py` 的调度路径,以及 `vllm/v1/spec_decode/` 下 ngram_proposer、medusa、metrics、llm_base_proposer、extract_hidden_states 等多个 proposer 相关文件(diff 文件清单),影响各类 SD 方法的 proposal 长度;正文注明"many ways to extend this strategy like resetting AR after some steps but those are left for future work",且 DSD 在 K=0 时仍有 draft prefill 开销。

## 关联

无(PR 正文仅引用 issue #4565 与 TurboSpec 论文,未给出其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
