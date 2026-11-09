# PR #13365: [V1][Spec Decode] Optimize N-gram matching with Numba

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2025-02-18 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, ci/build, v1 |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `ngram_proposer.py`、`requirements-common.txt`、`gpu_model_runner.py` 预热逻辑 |
| 链接 | https://github.com/vllm-project/vllm/pull/13365 |

## 一句话结论

用 Numba JIT 编译 V1 N-gram 提议者的 KMP 匹配,大批量下 CPU 开销 54.3s → 1.9s(20-30x)。

## 动机

V1 的 N-gram 投机解码在每个 engine step 都要对 batch 内每条序列做 n-gram 匹配(KMP),当时是纯 Python 静态方法实现,原代码里就有 TODO:"Use c++ to implement the _find_subarray_kmp to improve the efficiency"。大批量场景下这一 CPU 路径成为显著开销:ShareGPT 5K 请求的基准里,匹配累计耗时 54.3 秒,是整体运行时间的重要部分。

## 优化点

算法不变(KMP 找 pattern 在 context 中最后一次出现、返回其后 k 个 token),只做执行层优化:把 `_kmp_lps_array` 与 `_find_subarray_kmp` 从类静态方法改为模块级函数并加 `@jit(nopython=True)`,list 换成 `np.zeros(len, dtype=np.int32)` 以便 Numba 编译为原生循环;`requirements-common.txt` 钉住 `numba == 0.60.0`(v0.61 不支持 Python 3.9);`gpu_model_runner` 初始化 NgramProposer 时用 1024 长度的零数组立即触发一次 JIT 编译(<1 秒),避免首个请求承担编译延迟。PP 场景下 drafter 只在 last rank 创建。

## 关键改动

```text
vllm/v1/spec_decode/ngram_proposer.py: KMP 匹配改为 @jit(nopython=True)
  模块级函数,数据结构改 numpy 数组
vllm/v1/worker/gpu_model_runner.py: 初始化时预热 JIT;drafter 仅
  last rank 创建
requirements-common.txt: 新增 numba == 0.60.0
```

关键片段:

```python
@jit(nopython=True)
def _find_subarray_kmp(
    context_token_ids: np.ndarray, n: int, k: int,
) -> Optional[np.ndarray]:
    pattern = context_token_ids[-n:]
    lps = _kmp_lps_array(pattern)
```

## 性能声明

PR 正文:大批量下观察到 20-30x 提速;ShareGPT 5K 请求基准,N-gram 匹配累计开销从 54.3 sec 降到 1.9 sec,约占整体运行时间的 2.5%。PR 未注明 GPU/硬件型号(该开销为 CPU 侧)。

## 验证与回归风险

PR 正文未列测试改动;N-gram 匹配语义逐行保持(算法与返回值结构不变),`propose` 的 docstring 仅更新类型说明。风险:引入 numba 依赖(固定 0.60.0,与 numpy <2.0 约束共同生效),首次 JIT 编译被预热带走但仍是进程级一次性成本;`Optional[np.ndarray]` 返回类型在 nopython 模式下的行为依赖 numba 版本。

## 关联

V1 N-gram 投机解码路径;同分类:#5765(V0 bonus token)、#22872/#24505/#25916(EAGLE)、#23569(async scheduling 与投机解码集成)、#63801(HiSparse MTP 验证预取)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
