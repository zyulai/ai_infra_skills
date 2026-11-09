# PR #44785: [Fix] srt: Apply --gc-threshold in every server process and freeze every tokenizer worker

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#44785](https://github.com/sgl-project/sglang/pull/44785) |
| 作者 | metamergebot |
| 合并日期 | 2026-10-16 |
| labels | run-ci, bypass-fail-fast |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Frontend / Serving |
| bug 类型 | hang |

## 一句话结论
--gc-threshold 只在启动进程生效、多 tokenizer 只冻一个 worker;改为所有服务进程应用阈值并各自 freeze。

## 问题现象
正文描述:配置了 `--gc-threshold`(如调高 gen2 阈值以停止周期性 full collection)的部署,scheduler 与 detokenizer 子进程仍保持 Python 默认阈值,周期性 full collection 依旧发生在开销最大的 scheduler 里;多 tokenizer(`--tokenizer-worker-num`>1)时,post-warmup 的 `/freeze_gc` 请求由接收到它的那个 worker 处理,只有该 worker 的解释器被冻结,其余 worker 仍每次 collection 扫描长期堆。

## 根因
正文披露:
- `--gc-threshold` 只通过 `gc.set_threshold` 应用在启动(launching)进程,scheduler 与 detokenizer 子进程没有继承该配置;
- `/freeze_gc` 请求在多 tokenizer worker 下只被一个 worker 消费,本地 freeze 覆盖不全。

## 修复方案
- `common.py` 的 `freeze_gc` 增加可选 `gc_threshold` 参数,在 `gc.freeze()` 前 `gc.set_threshold(*gc_threshold)`,None 则不动阈值;
- `FreezeGCReq` 增加 `gc_threshold` 字段;`TokenizerManager.freeze_gc` 从 device args 读取阈值随请求下发并用于自身 freeze;scheduler 与 detokenizer 的 handler 把 `recv_req.gc_threshold` 透传;
- `http_server.py` 的 `_wait_and_warmup` 新增 `post_warmup_callback`(warmup 之后、路由 `/freeze_gc` 之前执行);多 tokenizer 模式下每个 worker 的 warmup 线程获得回调,本地按配置阈值 freeze 自己,路由请求仍覆盖 scheduler 与 detokenizer。默认行为不变:未配置阈值不新设,单 tokenizer 路径冻结点不变(正文声明)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:GC 相关路径——freeze_gc 工具函数、FreezeGCReq 序列化(新增字段)、tokenizer manager/scheduler/detokenizer 的 freeze 处理、http_server warmup 流程(新增回调点);仅影响配置了 --gc-threshold 或多 tokenizer 的部署,默认配置为 no-op。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
