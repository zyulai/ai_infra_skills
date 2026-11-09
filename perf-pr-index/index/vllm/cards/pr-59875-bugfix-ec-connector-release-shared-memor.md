# PR #59875: [Bugfix][EC Connector] Release shared memory on NIXL setup failure

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @DGlan |
| 合并 | 2026-10-09 |
| 分类 | Overlap / Scheduler(tier: major) |
| Label | bug, ready, cpu, kv-connector |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/59875 |

## 一句话结论

修复 ECCPU scheduler 构造函数在 NIXL 初始化失败时泄漏已分配共享内存(mmap 文件、fd、映射)的资源清理缺陷。

## 动机

EC(EC/Embedding Cache)connector 的 CPU scheduler 构造函数先分配共享内存区域(`create_ec_shared_region`),再调用 `_setup_nixl()` 完成 NIXL 数据通道注册与 ZMQ 控制通道绑定。若后者抛异常(如未安装 NIXL、control transport 创建失败),异常直接从构造函数传播,调用者永远拿不到 scheduler 实例、无从调用 `shutdown()`——已创建的 mmap 文件、两个文件描述符与内存映射全部泄漏。启动失败可反复重试,泄漏随之累积。

## 优化点

把失败清理的职责收拢到构造函数:`__init__` 中 `_setup_nixl(vllm_config)` 包上 `try/except Exception: self.shutdown(); raise`,复用正常关闭路径统一释放 NIXL 资源与共享区域,并保留原异常语义。相应地删除 `_setup_nixl()` 内部针对"注册 NIXL + 绑定控制 socket"段的局部 `try/except → _teardown_nixl()` 处理器,该函数现在只负责传播错误,避免双重清理;transport/session 创建逻辑仅做等价重排(NixlDataTransport、ProducerSession、ZmqClientTransport 的构造顺序与参数不变)。净效果是任何一步失败都走同一条 unwind 路径,无推理行为或输出变化。

## 关键改动

```text
vllm/distributed/ec_transfer/ec_connector/cpu/scheduler/__init__.py: 构造函数 try/shutdown/raise;移除 _setup_nixl 内部清理
tests/v1/ec_connector/unit/test_scheduler_nixl_ctor.py: 新增两个失败清理回归测试
```

核心修复(节选):

```python
if self._nixl_enabled:
    try:
        self._setup_nixl(vllm_config)
    except Exception:
        self.shutdown()
        raise
```

## 性能声明

无任何性能声明(regex 检测为 none);PR 明确写明 "No inference-output or performance change is claimed."。价值在于消除启动失败时的共享内存与 fd 泄漏(如容器内反复重启场景下防止内存累积)。验证环境:Ubuntu 24.04/WSL,Python 3.12.14,torch 2.13.0+cpu,vLLM v0.30.0 CPU 预编译二进制,NIXL 1.4.1。PR 未注明其余测试条件。

## 验证与回归风险

新增两个 `@pytest.mark.cpu_test` 回归测试:`test_missing_nixl_releases_shared_region`(NIXL 缺失时断言 mmap 文件不存在、`_fd`/`_mmap_obj` 为 None,异常消息 "ec_enable_nixl requires NIXL" 保留)、`test_control_transport_failure_releases_nixl_and_shared_region`(data transport 注册成功后 control transport 失败,断言 NIXL 注销、共享区域释放、原异常对象原样抛出)。另有独立 CPU 检查经 `ECConnectorFactory`/`ECCPUConnector` 用真实 VllmConfig 与本地 tiny Llava 配置跑全流程,通过 `/proc/self/fd` 与 `/proc/self/maps` 核对:修复前泄漏 1 文件 + 2 fd + 1 映射,修复后为零。套件结果:constructor 测试 7 个全过(修复 import 顺序导致的测试间污染:先 import `data.nixl` 再 monkeypatch,避免缓存缺失 NIXL 的绑定)、相关传输/会话/scheduler 95 个测试通过、CPU EC 全套 62 passed/350 deselected,pre-commit(含 mypy)通过。限制:未跑完整 GPU serving 与跨进程 NIXL 数据传输,验证覆盖构造与启动失败清理。回归风险很小——改动面限于失败路径;正常启动路径的 transport 创建语句被重排但语义等价。测试被 `.buildkite/test_areas/misc.yaml` 的既有 CPU EC job 选中,CI 有覆盖。

## 关联

Related: #56507、#56508(相关 test/doc PR,覆盖 import 隔离与文档,非构造清理);PR 作者声明未发现同区域等效开放 PR。本批次内同分类无其他卡。
