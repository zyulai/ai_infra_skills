# PR #23305: [misc] CI hygiene: enforce __main__ entry, drop silent-skipped tests, fix rerun-test protoc

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23305](https://github.com/sgl-project/sglang/pull/23305) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-04-21 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
CI 用 `python3 file.py -f` 跑测试,pytest 风格测试文件缺 `__main__` 块时只导入即退出 0,测试从未真正运行;新增 AST 级 sanity check 强制入口、补齐 6 个文件并删 9 个静默跳过文件,另修 rerun-test 缺 protoc/rustup。

## 问题现象
`test/registered/` 下的 pytest 风格测试在 CI 中被静默跳过:CI runner 通过 `python3 file.py -f`(而非 pytest)调用每个文件,没有可执行 `if __name__ == "__main__":` 块的文件只是导入后以 0 退出——每个测试函数体从未运行,CI 却显示绿色。直接触发点是 streaming session 单测(`_FakeReq` 缺 `finished_len = None` 字段,真实 `Req` 默认为 None)。另外 `rerun-test-cpu` 工作流缺 protoc/rustup 步骤:安装 sglang editable wheel 需编译 `sglang-grpc` Rust crate,其 `build.rs` 需要 protoc,导致 workflow 在 `Install dependencies` 阶段就失败。

## 根因
正文披露:CI 执行链 `run_unittest_files` -> `subprocess.Popen(["python3", <file>, "-f"])` 假定 `unittest.main()` 会把 `-f` 当作 `--failfast`;pytest 风格文件没有 `unittest.main()`/`__main__` 块时只是导入 + exit 0,零测试执行。跨目录迁移测试时容易在 review 中悄悄混入。rerun-test.yml 部分:`runs-on: ubuntu-latest` 但缺少 pr-test.yml 的 `stage-a-test-cpu` 所有的 `Install protoc` + `Install Rust toolchain` 两步。

## 修复方案
1. `python/sglang/test/ci/ci_register.py`:`RegistryVisitor` 新增 `has_main_entry` 与 AST 检查 `_is_main_block_with_call`(识别 `if __name__ == "__main__":` 且体内至少含一个 call),`ut_parse_one_file` 改为返回 `(registries, has_main_entry)`;`collect_tests` 增加 sanity check:至少有一个启用 registry 的注册文件必须有可执行 `__main__` 块,全部 `disabled=...` 的辅助文件豁免。调用方 `scripts/ci/check_registered_tests.py`、`ci_coverage_report.py` 同步适配。关键 diff:

```python
+    @staticmethod
+    def _is_main_block_with_call(stmt: ast.If) -> bool:
+        """True iff `stmt` is `if __name__ == "__main__":` with a body that
+        contains at least one call ..."""
+        for child in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
+            if isinstance(child, ast.Call):
+                return True
+        return False
```

2. 补 `__main__` 入口保留 6 个文件(streaming_session_unit、debug_utils 5 个,并删除 `TestMainBasic` 子集——其 3 个集成测试调 `dump_comparator.main()` 硬编码 `x.cuda()`,在 CPU 套件必然报 no NVIDIA driver);删除 9 个静默跳过的文件(test_bnb、test_int4fp8_moe、test_zimage_turbo、jit_kernel test_dependency、test_repeat_interleave、test_glm47_moe_detector、test_model_hooks、test_mrope、test_patch_embed_perf)。
3. `.github/workflows/rerun-test.yml`:镜像 pr-test.yml,新增 `Install protoc` 与 `Install Rust toolchain` 两步。
另删过时测试 `defers_tail_free`(意图已被 #22897 取代)。注:pack 的 diff 在 40000 字符处截断,后半部分(部分测试文件的增删内容)未展示。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:CI 侧——sanity check 会让今后无 `__main__` 入口的注册测试无法合入;9 个被删文件的测试不再执行(相关覆盖消失,作者需带 `__main__` 块重开);6 个保留文件从此真正在 CI 运行(可能暴露此前从未跑过的问题);`ut_parse_one_file` 返回值变更影响所有调用方。rerun-test.yml 增加两个安装步骤。

## 关联
- [#22897](https://github.com/sgl-project/sglang/pull/22897)(取代 defers_tail_free 测试)
- [#16285](https://github.com/sgl-project/sglang/pull/16285)(test_dump_comparator.py 来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
