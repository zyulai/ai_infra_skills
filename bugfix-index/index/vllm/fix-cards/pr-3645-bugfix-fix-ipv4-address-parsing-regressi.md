# PR #3645: [BugFix] Fix ipv4 address parsing regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3645](https://github.com/vllm-project/vllm/pull/3645) |
| 作者 | njhill |
| 合并日期 | 2024-03-26 |
| labels | (none) |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#3641 为支持 IPv6 无条件给 IP 加方括号,导致 IPv4 下 torch.distributed 初始化报错;改为仅 IPv6 加括号。

## 问题现象
使用 IPv4 地址做分布式初始化时,worker 启动即抛异常:`ValueError: An IPv4 address cannot be in brackets`。堆栈路径为 `worker.py init_device` → `init_distributed_environment` → `torch.distributed.init_process_group` → `rendezvous` → `urlparse/urlsplit` → `_check_bracketed_host`。触发条件:分布式初始化地址被格式化为 `tcp://[<IPv4>]:<port>`,而 Python 3.11 的 urlsplit 校验拒绝方括号中的 IPv4。

## 根因
正文披露:#3641 是针对 IPv6 的修复,但破坏了 IPv4 场景——`get_distributed_init_method` 把所有 IP 统一格式化为 `tcp://[{ip}]:{port}`,而 IPv4 地址按规范不允许包裹方括号,rendezvous 解析 URL 时直接抛 ValueError。

## 修复方案
`vllm/utils.py` 的 `get_distributed_init_method`:仅当 IP 中含冒号(即 IPv6 地址)时才加方括号,IPv4 使用裸地址:

```diff
 def get_distributed_init_method(ip: str, port: int) -> str:
-    return f"tcp://[{ip}]:{port}"
+    # Brackets are not permitted in ipv4 addresses,
+    # see https://github.com/python/cpython/issues/103848
+    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"
```

## 影响与回归风险
- 影响窗口:#3641 合并之后至本 PR 之前,以 IPv4 地址进行 torch.distributed 初始化的部署(单机多卡与多机均经此路径)。
- 回归面:`get_distributed_init_method` 的返回格式,是分布式 rendezvous 地址的唯一来源;含冒号的 IPv6 地址继续使用方括号,行为不变;IPv4 恢复为 #3641 之前的裸地址格式。改动为单行字符串格式化,无其他调用面。

## 关联
- PR #3641(引入回归的 IPv6 修复)
- diff 注释引用 python/cpython#103848(IPv4 不允许方括号的依据)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
