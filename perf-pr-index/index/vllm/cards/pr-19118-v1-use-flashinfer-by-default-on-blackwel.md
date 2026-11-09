# PR #19118: [V1] Use FlashInfer by default on Blackwell GPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-06-05 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/19118 |

## 一句话结论

在 Blackwell(SM 10.0)GPU 上默认选择 FlashInfer attention 后端,替代 V1 中性能明显较差的 FlashAttention2 默认路径。

## 动机

V1 引擎在 CUDA 上的默认 attention 后端是 FlashAttention,但 FlashAttention3 不支持 Blackwell,实际落到 FA2;而 FlashInfer 有专门针对 NVIDIA Blackwell 的 backend,性能显著更好。作者认为只要用户装了 flashinfer 就应该默认选择它,基准数据见 PR #18095 的 benchmark 评论。

## 优化点

改动很小且机制清晰:在 `get_attn_backend_cls` 的后端选择逻辑中,当设备 compute capability 精确等于 SM 10.0 时,先尝试 `import flashinfer`,成功即返回 `FlashInferBackend`;导入失败则打日志提示安装 flashinfer 可获得更好性能,并继续落到原有 FlashAttention 分支。为支持"精确匹配某一代 cc"(而非 `has_device_capability` 的大于等于语义),在 platform 接口新增 `is_device_capability`,支持 `(major, minor)` 元组或整数两种入参。用户显式指定后端时优先级不变(该分支位于显式选择/饱和判定之后)。

## 关键改动

```text
vllm/platforms/cuda.py: get_attn_backend_cls 对 SM 10.0 先试 FlashInfer,失败回退 FlashAttention
vllm/platforms/interface.py: 新增 is_device_capability 精确 cc 匹配接口
```

核心选择逻辑:

```python
if cls.is_device_capability(100):
    # Prefer FlashInfer for V1 on Blackwell GPUs if installed
    try:
        import flashinfer  # noqa: F401
        return ("vllm.v1.attention.backends."
                "flashinfer.FlashInferBackend")
    except ImportError:
        pass
```

## 性能声明

PR 正文本身未给出 benchmark 数字,性能依据是引用 PR #18095 中评论 https://github.com/vllm-project/vllm/pull/18095#issuecomment-2877849390 的 benchmark(声明 FlashInfer 在 Blackwell 上"much more performant than FlashAttention2");Test Result 仅展示后端选择日志(B200 上有/无 flashinfer 的两种默认路径),非性能数字。具体加速倍数 PR 未注明测试条件。

## 验证与回归风险

验证:在 B200 本地测试两种情况——未安装 flashinfer 时回退 FlashAttention 并输出建议安装日志;安装后默认选中 FlashInferBackend。均通过 `vllm serve facebook/opt-125m --enforce-eager` 冒烟。回归风险:只改变 SM 10.0 且未显式指定后端时的默认值,非 Blackwell 设备与显式 `--attention-backend` 用户不受影响;隐性风险是 Blackwell 用户环境里 flashinfer 版本兼容性成为默认路径的一部分,导入成功但运行异常的场景由 FlashInferBackend 自身负责。

## 关联

基准来源 PR #18095;同区域相关:索引中 FlashInfer 后端系列卡(如 pr-20034 FlashInfer ragged prefill、pr-21078 trtllm-gen MLA decode)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
