# RFC #65707: [RFC]: [Draft] Split FusedMoEQuantConfig into static schema and runtime state

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65707](https://github.com/vllm-project/vllm/issues/65707) |
| 作者 | bnellnm |
| 创建日期 | 2026-11-04 |
| 状态 | 🟢 open |
| labels | RFC, quantization |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |

## 一句话结论
提议把 FusedMoEQuantConfig 拆为不可变无张量的静态 schema 与仅含已加载张量的运行时 config,让 MoE kernel 在权重加载前即可构建。

## 提案动机
`FusedMoEQuantConfig` 目前混装两类信息:静态信息(量化 dtype、group shapes、对称性、scale 布局、激活常量及 `is_block_quantized`/`use_fp8_w8a8` 等派生谓词)与运行时信息(weight scales、activation scales、global scales、zero points、biases、Triton `PrecisionConfig`,某些路径甚至 weights 本身)。由此带来一连串问题:kernel 构建依赖已加载张量(FlashInfer CuteDSL 读 scale 衍生 buffer、TRT-LLM 做 scale 处理、emulation experts 重写 quant config → kernel 必须等 load 完才能建);post-load 顺序隐式且脆弱(有的 expert 要求 kernel 先存在,有的要先变换 scale,有的导致 kernel 重建/替换,不相关对象间形成顺序依赖);quant config 是可变共享状态(有 expert 构造后清 scales/改 effective dtype,同一 config 在 hook 前后含义不同,使 weight reload、CUDA graph capture 与"捕获 kernel 会读哪些张量"的推理复杂化);emulation 后端可能按 checkpoint scheme 选择却执行另一 scheme,如今靠变更 config 表达;oracles、quant methods 与 kernel 类三处重复推导同一信息;backend 选择与 kernel 构建被拆在 model setup 两端,顺序是偶然的。

## 提出的方案
- 拆为两个生命周期不同的对象:`FusedMoEQuantSchema` 是不可变、无张量的量化方案静态描述(由 quantization method 持有),含 a1/a2/w1/w2 四个 role 的 `QuantKey`、不可从 key 派生的 role metadata(`has_bias`/`has_zero_point`/`alignment_shape`)、GEMM1 激活参数(`gemm1_alpha/beta/clamp_limit`)及 backend 选定的 scale 布局字段(`is_scale_swizzled`/`is_scale_transposed`/`mx_alignment`);`FusedMoEQuantConfig` 保持为运行时对象,只含加载的 weights、scales、zero points、biases 等张量,迁移期临时持有静态引用、之后移除。
- 外部 API 经 property 保持不变:调用方继续使用 `quant_dtype`、`block_shape`、`a1_scale`、`w1_bias` 等现有名字;dataclass 字段私有,跨模块只走 `@property`/显式方法,禁止直读 `_schema`、`_a1`/`_w1`。
- 确立唯一顺序:`quant method __init__`:静态 schema keys → 选 backend → 构造 kernel(kernel 构造器 finalizes schema ABI);`weight loading`:变换 weights → 构建运行时 config → 执行时显式传入。kernel 与 experts 不拥有、不缓存 runtime config;post-load 处理不再选 backend、不再构造 kernel、不再变更 quantization schema;不新增 setup hook,`RoutedExperts.__init__` 序列不变。
- `FusedMoEStaticQuantDesc`(frozen dataclass)描述单 role:key、可选 alignment_shape(不得与 key.scale.group_shape 重复)、has_bias/has_zero_point;schema 的 a1/a2、w1/w2 角色分离使未来混合 scheme 无需改 schema 类型(现有 kernel 多要求 a1==a2、w1==w2,初始 builder 以同一 key 填两角色维持该约束)。
- Oracle 选择只消费 schema 的 role keys(`allow_vllm_cutlass`、用户覆盖、平台检查等是 oracle 显式参数,不是 schema 字段);选择不得检视 ABI 字段;backend 选定后由 kernel 构造器构造新的不可变 schema 值完成 ABI finalization 并存为 kernel canonical schema。single-key 适配器经 `activation_key`/`weight_key` 属性取 common key(a1!=a2 或 w1!=w2 时 raise)。
- Emulation 后端的反量化是所选 expert 的执行策略而非第二种 quantization scheme:schema 保留 int4 weight keys,expert 反量化运行时张量并走 unquantized 计算路径,不得清 schema 字段或构造期改共享 runtime config。
- `FusedMoEMethodBase` 新增 `fused_moe_quant_schema()`;现有 `get_fused_moe_quant_config` 重用为 runtime builder(名字、签名、可选返回不变,非 None 结果必须是完整最终张量 payload,是唯一允许收集该 payload 的方法)。
- 运行时 config 在全部 weight/scale 变换完成后构建,构建时校验张量 presence/dtype/shape 与静态契约一致(shape 校验须 backend-aware,swizzled/transposed scales 的物理布局可异于逻辑 QuantKey shape);建成后视作不可变(reload/权重替换路径构建新对象而非原地改字段);kernel/experts 每次调用显式接收,不得跨调用保留;无 weights 的 runtime config 不是 modular kernel 的合法输入,仅需静态信息的单测/基准必须用 `FusedMoEQuantSchema`。
- 分阶段独立可评审落地:Stage 1 在现有 API 之后引入静态 config(纯增量),Stage 2a-3 起增量迁移内部消费者并推迟构造器/工厂签名变更;`_schema` 兼容引用在所有调用方直用 schema 后移除。

## 优化点与收益
正文未给出量化数字。收益为工程性:modular MoE kernel 的构建时机从「长且随实现而异的 post-load 序列末端」提前到各 quantization method 的 `__init__`(权重加载前);消除 post-construction 对量化 scheme 字段的变更;oracle 选择与 kernel 构造可在不分配真实量化权重的情况下测试;以单一权威静态描述消除 oracle/quant method/kernel 的重复推导。

## 现状与落地
🟢 open(2026-11-04 创建;正文自标 Status: Proposal/RFC、Date 2026-11-03),0 评论,处于早期征求意见阶段;正文给出 API 兼容规则与分阶段迁移计划。注:pack 正文在 20k 字符处截断(「Kernel and oracle APIs」一节 `make_kernel` 签名处中断),本卡基于现有内容生成。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
