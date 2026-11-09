# PR #2714: [Fix] fix incorrectly overwriting the port specified in ServerArgs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2714](https://github.com/sgl-project/sglang/pull/2714) |
| 作者 | mickqian |
| 合并日期 | 2025-01-02 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
服务启动时端口预分配从固定 10000 扫起、错误覆盖 ServerArgs 指定的 port;改为从用户指定的端口开始扫描。

## 问题现象
正文仅以「Addressing #2430」引出,未描述具体现象。从 diff 看,问题出在 `python/sglang/srt/server.py` 启动时的端口预分配逻辑:扫描循环起点硬编码为 10000,循环结束后执行 `self.server_args.port = port`,把用户在 ServerArgs 中指定的端口覆盖为从 10000 起找到的可用端口(由 diff 推断)。

## 根因
正文未披露根因,只说明修改意图是「Pick up the port specified in `ServerArgs`」。(由 diff 推断)端口预分配写成 `for port in range(10000, 40000)`,完全不参考 `self.server_args.port` 中用户指定的值;只要 10000 起有可用端口,用户指定值就必然被覆盖。原循环体内还有一句 `port += 1`,在 `for ... in range(...)` 的步进之外多做一次自增,属冗余/干扰代码。

## 修复方案
`python/sglang/srt/server.py`:预分配循环改为从 `self.server_args.port` 起扫描,并删除循环体内多余的 `port += 1`:

```diff
-        for port in range(10000, 40000):
+        for port in range(self.server_args.port, 40000):
             if is_port_available(port):
                 break
-            port += 1
         self.server_args.port = port
```

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:改动仅涉及 `server.py` 启动期的端口预分配。行为变化:指定端口被占用时,从该指定端口向上继续寻找可用端口,而不是回到 10000 起重新扫;扫描上界仍为 40000。(由 diff 推断)若用户指定的端口 ≥ 40000,`range` 为空、循环不执行,此处行为与旧版不同,值得关注。

## 关联
- [#2430](https://github.com/sgl-project/sglang/issues/2430)(本 PR 的 Motivation 即为解决该 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
