# RFC #45585: [RFC] RL CI Matrix for vLLM: Behavioral + Physical + Protocol Coverage

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45585](https://github.com/vllm-project/vllm/issues/45585) |
| 作者 | aoshen02 |
| 创建日期 | 2026-06-14 |
| 状态 | ⚪ closed (关闭于 2026-09-08) |
| labels | RFC |
| 热度 | 👍 6 · 💬 6 |
| 分类 | Other |

## 一句话结论
提出面向 RL 场景的结构化 CI 矩阵,覆盖 sleep/wake、pause/resume 与权重传输 HTTP 协议这一零 CI 覆盖的生产 bug 高发面。

## 提案动机
vLLM 是 verl、slime、OpenRLHF、SkyRL 等 12 个后训练框架的主流 rollout 引擎;RL 生命周期涉及独特代码面:cumem GPU 内存管理(sleep/wake_up)、scheduler gating(pause/resume)、权重传输 HTTP 协议(/start_weight_update → /update_weights → /finish_weight_update)。该面近月爆发一批生产 bug(6 个 open PR:#45326、#45552、#45363、#45518、#44972、#44483,如 sleep() 缺 torch.cuda.synchronize() 导致「200 lie」),但今天没有任何专属 CI 覆盖;现有测试只断言 flag 值与 Prometheus 指标,从不验证行为后果(如 scheduler 是否真的阻塞请求)。

## 提出的方案
- 在 tests/entrypoints/serve/dev/rlhf/ 下按 state_transitions/、info_retrieval/、consistency/ 组织测试,每个文件经 parametrize 同时覆盖单卡与 TP≥2
- 覆盖面:sleep/wake 生命周期与 CPU 状态机不变量(作者称行业空白:无框架做 CPU 级状态机测试)、PauseMode abort/wait/keep、权重传输协议 framing 与错误路径、SHA-256 checksum、Prometheus 指标、/weight_info 与 /update_weight_label 版本端点、结构化 JSON 响应
- 每个测试场景标注参考的下游框架(sglang、ROLL、trl、AReaL、miles、verl 等)与参考测试文件
- 把散落各处的 RL 测试整合进 rlhf/ 目录;conftest.py 共享设施(ServerContext、wait_for_health 等)参考 miles 与 sglang
- 工作量估计约 25 个新测试;PR #45586 已写入 test_sleep_wake.py(26 tests)与 test_weight_update.py(11 tests)

## 优化点与收益
照抄数字(实施进度记录):Phase 0-2 完成——conftest + test_state_machine.py 16 tests、test_sleep_wake.py 28 total、test_weight_update.py 16 total、test_multi_gpu.py(TP=2)4 tests;Phase 1 使 /sleep、/wake_up 返回结构化 JSON;Phase 2 新增 /weight_info 与 /update_weight_label。E2E 验证(GB200 NVL 容器,aarch64,Python 3.12,vLLM 0.23.0):test_state_machine 16/16 通过(8m42s)、test_response_schema 5/5(3m05s)、test_weight_info 8/8(4m35s),test_sleep_wake 结果在截断处中断。

## 现状与落地
已关闭。正文记录 Phase 0-2 全部完成并有 E2E 验证;因 pack 正文在 20k 字符处截断(E2E 验证表格中断),最终落地情况需查关联 PR(如 #45586)。

## 关联
- PR #45586(测试文件):https://github.com/vllm-project/vllm/pull/45586
- #46438:https://github.com/vllm-project/vllm/pull/46438
- #46234:https://github.com/vllm-project/vllm/pull/46234
- #44972:https://github.com/vllm-project/vllm/pull/44972
- SkyRL#1846:https://github.com/NovaSky-AI/SkyRL/issues/1846

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
