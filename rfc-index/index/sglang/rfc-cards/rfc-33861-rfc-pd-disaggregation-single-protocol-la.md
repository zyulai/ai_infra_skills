# RFC #33861: [RFC] PD disaggregation: single protocol layer, per-backend transport

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33861](https://github.com/sgl-project/sglang/issues/33861) |
| 作者 | hnyls2002 |
| 创建日期 | 2026-08-06 |
| 状态 | ⚪ closed (2026-10-15) |
| labels | inactive |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Disaggregated Serving |

## 一句话结论
让 PD 传输协议(bootstrap+transfer 控制)只存在一份,mooncake/nixl/mori 三个后端退化为纯 transport 实现。

## 提案动机
三个 PD 传输后端(mooncake、nixl、mori)各自独立实现整套 bootstrap + transfer 控制协议:ZMQ socket 循环、消息分发、room/状态生命周期、staging 胶水、TP/DCP 切片编排、超时与失败处理。三者描述同一协议却已漂移,且互有缺失:mori 的 decode manager 从不启动心跳线程,死掉的 prefill peer 永远检测不到(mooncake 与 nixl 都启动);nixl 的 sender 不执行 bootstrap 超时,prefill 侧传输失败不传播到 decode——请求挂到通用等待超时而不是快速失败;transfer-info 消息两帧位置在 mooncake 与 nixl 间互换,两帧都是整数,错配会静默交换两个字段而非干净报错。每个协议修复或特性最多要写三遍:staging 与 DCP relayout 只在 mooncake+nixl 有,session 黑名单只在 mooncake 有。

## 提出的方案
- 目标终态:一套具体协议类 KVManager / KVSender / KVReceiver / KVBootstrapServer,无后端子类;socket 循环、解析分发、room 生命周期、staging、chunking、TP/DCP 切片数学、超时、心跳、失败传播都在其中实现一次。
- 后端只提供构造时注入的 transport 对象:BaseTransport 接口(init_engine / register_memory / local_desc / open_peer / submit / poll);协议层决定哪些字节去哪、何时去,transport 只搬运 (src_ptr, dst_ptr, len) 并命名远端内存。
- get_kv_class / KVClassType 枚举表换成 typed 的 make_transport(backend);fake 后端变 null transport,使协议层可在 CPU-only CI 中测试。
- 非大爆炸重构:走许多小 PR——可独立评审的行为对齐修复 + 已相同代码的机械上提;wire-format 统一放最后,且因今天没有版本协商,需要独立迁移方案。
- 作者明确这是 early proposal:接口是草图,终态能否完全按图画出来是开放的;主要诉求是 mooncake/nixl/mori 三个后端 owner 的反馈。

## 优化点与收益
正文未给出量化数字。定性收益:消除三份实现间的漂移与三重修复成本(当前 mooncake 调 5 个引擎方法、nixl 14 个、mori 8 个)。

## 现状与落地
已关闭(2026-10-15),labels 为 inactive,属早期提案未推进完即关闭;正文未列出任何承接 PR,落地情况需查关联 PR/后续讨论。

## 关联
正文未列出具体链接(仅按名提及 mooncake、nixl、mori、fake 后端)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
