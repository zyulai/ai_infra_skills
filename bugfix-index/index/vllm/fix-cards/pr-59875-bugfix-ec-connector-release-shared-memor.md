# PR #59875: [Bugfix][EC Connector] Release shared memory on NIXL setup failure

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59875](https://github.com/vllm-project/vllm/pull/59875) |
| 作者 | DGlan |
| 合并日期 | 2026-10-09 |
| labels | bug, ready, cpu, kv-connector |
| 热度 | 👍 0 · 💬 28 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
ECCPU scheduler 构造中 NIXL setup 失败时已分配的共享内存区域泄漏,构造函数改为异常时经 shutdown() 释放资源并保留原异常。

## 问题现象
启用 NIXL 时,ECCPU scheduler 初始化失败会泄漏共享内存:共享区域(shared region / mmap 文件、文件描述符、内存映射)在 `_setup_nixl` 之前就已分配;若 `_setup_nixl` 抛异常,构造函数直接向外传播,调用方永远拿不到 scheduler 实例、也就无从调用 shutdown 释放,已分配的 mmap 文件、fd 和映射全部残留。作者用 `/proc/self/fd` 与 `/proc/self/maps` 实测确认:修复前缺失 NIXL 的错误路径留下 mmap 文件、两个文件描述符和一个映射;修复后同样报错但残留为零。

## 根因
正文披露根因:共享内存区域在 `_setup_nixl` 之前分配,`_setup_nixl` 内部的 try/except 只清理 NIXL 资源(`self._teardown_nixl()`)而不释放共享区域,随后原样 raise;由于失败发生在构造函数内,调用方拿不到对象,没有任何人负责释放 region,造成共享内存泄漏。即构造失败时的资源回收路径不完整——缺少从构造函数自身出发、经 `shutdown()` 的完整失败回退(failure unwinding)。

## 修复方案
改 `vllm/distributed/ec_transfer/ec_connector/cpu/scheduler/__init__.py`:
1. `__init__`:把 `self._setup_nixl(vllm_config)` 包进 try/except,捕获任何异常时先 `self.shutdown()` 再 `raise`,构造函数拥有失败回退职责,先释放 NIXL 资源再释放共享区域,并保留原异常。
2. `_setup_nixl()`:删除内部「注册 region 到 NIXL + 绑定 control socket」段落的 try/except 与 `self._teardown_nixl()` 二次清理,让错误直接向上传播(避免双重清理),`NixlDataTransport`/`ProducerSession`/`ZmqClientTransport` 的构建逻辑本身不变。
3. 测试 `tests/v1/ec_connector/unit/test_scheduler_nixl_ctor.py` 新增两个 cpu_test:`test_missing_nixl_releases_shared_region`(缺 NIXL 时断言 mmap 文件不存在、fd 与 mapping 为 None)和 `test_control_transport_failure_releases_nixl_and_shared_region`(control transport 创建失败时断言 NIXL 已注销、共享区域已释放、原异常对象保留)。

关键 diff 摘录:
```diff
         if self._nixl_enabled:
-            self._setup_nixl(vllm_config)
+            try:
+                self._setup_nixl(vllm_config)
+            except Exception:
+                self.shutdown()
+                raise
...
-        try:
-            self._data = NixlDataTransport(
...
-        except Exception:
-            self._teardown_nixl()
-            raise
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;影响启用 NIXL 的 ECCPU(EC connector)scheduler 构造失败路径。
- 回归面:仅 ECCPU scheduler 的构造/析构失败回退;正常启动路径(注册、会话建立)逻辑未变。作者验证:NIXL 1.4.1 下构造器测试 7 个全过、相关 EC connector 单测 76 过、CPU 标记单测 62 过(350 deselect);显式声明不涉及推理输出或性能变化。`shutdown()` 现在可能被构造失败路径调用,其幂等性与部分初始化状态下的行为是主要依赖点(测试覆盖了缺 NIXL 与 control 失败两种形态)。

## 关联
- [#56507](https://github.com/vllm-project/vllm/pull/56507)(正文 Related 引用)
- [#56508](https://github.com/vllm-project/vllm/pull/56508)(正文 Related 引用,相关 test/doc PR 覆盖 import 隔离与文档)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
