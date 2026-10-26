# BRPA 模块原理与源码说明

本文档严格对应本次上传的 **BRPA Python 源码**。另外上传的 LaTeX 文件属于 TCPELIA 研究，因此不能将该论文中的论证与结果直接作为 BRPA 的依据。

## 1. 输入特征与分支构成

设输入特征为 $X\in\mathbb{R}^{B\times C\times H\times W}$，其中 $B$、$C$、$H$ 和 $W$ 分别表示批次大小、通道数、特征图高度与宽度。

**种子预测分支**：对输入特征依次执行 `3×3` 逐通道卷积、批归一化、SiLU、`1×1` 卷积和 Sigmoid，得到初始种子图 $S_0\in[0,1]^{B\times1\times H\times W}$。

**边界预测分支**：采用与种子分支相同的算子结构，但具有独立的网络参数，输出 $B\in[0,1]^{B\times1\times H\times W}$。代码将其作为传播屏障，并没有单独实现使用边界真实标签的监督项。

**通道调制分支（可选）**：利用全局平均池化、两层 `1×1` 卷积、ReLU 和 Sigmoid 计算 $C_w\in[0,1]^{B\times C\times1\times1}$；中间通道数为 $\max(\lfloor C/8\rfloor,16)$。

## 2. 边界引导区域传播

在第 $t$ 次迭代中，首先通过空间传播算子 $\mathcal{P}$ 计算邻域候选响应：

$$
\Delta_t=\operatorname{ReLU}\left(\mathcal{P}(S_t)-S_t\right).
$$

随后使用传播率 $p$ 和边界门控 $G(B)$ 更新种子响应：

$$
S_{t+1}=\operatorname{clip}\left(S_t+p\,G(B)\odot\Delta_t,0,1\right).
$$

其中 $\odot$ 表示逐元素乘法，标量及单通道映射按维度广播。可学习模式使用 $p=\sigma(\eta)$，并将 $\eta$ 初始化为 $0.5$，因此其**初始有效传播率约为 0.622459**，不是训练后的固定值。

### 可选传播算子

| 选项 | 源码中的实际实现 |
|---|---|
| `maxpool3` | 步幅为 1、填充为 1 的 `3×3` 最大池化 |
| `avgpool3` | 步幅为 1、填充为 1 的 `3×3` 平均池化 |
| `maxpool5` | 步幅为 1、填充为 2 的 `5×5` 最大池化 |
| `conv3` | 可学习 `3×3` 单通道卷积；候选响应为 `max(S, conv3(S))` |

对于最大池化，局部峰值可以向邻域延展；`ReLU` 限制仅对正向扩张增量进行更新。对于平均池化、可学习卷积等消融算子，负向增量同样不会引起当前传播图的衰减。

### 可选边界约束

| 选项 | $G(B)$ | 作用 |
|---|---|---|
| `none` | $1$ | 无传播屏障 |
| `soft` | $1-B$ | 根据边界预测值连续抑制传播 |
| `hard` | $\mathbb{I}(B<\tau)$ | 使用阈值生成二值屏障，不可对阈值运算反向传播 |
| `shifted` | $1-\operatorname{Roll}(B)$ | 将边界图循环平移，用于屏障空间对应关系消融 |
| `power` | $(1-B)^\gamma$ | 调整屏障抑制强度 |

以上变量和具体模式与源代码一致。没有对应实验结果的设置仅说明实现方式，不宣称其具备性能提升。

## 3. 空间细化与重校准

若启用空间细化，将传播结果 $S_K$、原始种子 $S_0$ 与边界图 $B$ 沿通道维拼接，采用一个 `3×3` 卷积及 Sigmoid 获得空间权重：

$$
A=\sigma\left(\operatorname{Conv}_{3\times3}([S_K,S_0,B])\right).
$$

若不启用细化，则 $A=S_K$。根据重校准模式得到不同形式的输出：

| 选项 | 输出表达式 |
|---|---|
| `spatial` | $Y=A\odot X$ |
| `spatial_channel` | $Y=A\odot X\odot C_w$（必须启用通道分支） |
| `spatial_residual` | $Y=X+A\odot X$ |
| `full` | 有通道分支时 $Y=X+A\odot X\odot C_w$，否则 $Y=X+A\odot X$ |

注意：`COMPONENT_VARIANT="full"` 表示启用完整组件；`RECALIBRATION_MODE="full"` 表示采用某种输出表达式，二者不是同一个控制变量。**仅仅将重校准模式设为 `full`，不意味着通道分支已启用。**

## 4. 使用与实验记录注意事项

- 上传版本的默认参数为 `COMPONENT_VARIANT="seed_prop_boundary_refine"`、`PROP_STEPS=4`，并**未**启用通道分支；本仓库刻意保持这一默认行为。
- 使用 `BRPA(channels=64, steps=2)` 将当前实例的迭代次数设为 2；不会修改其他实例或源代码顶层常量。
- `hard` 屏障采用阈值比较，对边界预测的该门控运算不可微，不应将其描述为与软屏障完全等价的训练机制。
- `get_brpa_ablation_tag()` 源码中使用全局设置拼接名字。当模块实例显式修改 `steps`，或者选择 `full` 组件但 `ENABLE_CHANNEL=False` 时，标签可能无法体现真实设置。建议同时记录 `print(model)`、参数配置与 `model.get_effective_p()`。
- 已有代码可以计算正向输出与反向梯度，但不包含数据集、YOLO 检测器改动、真实训练日志或模型权重。因此项目未提供未经核实的精度、实时性能或通用性结论。

## 5. 相关文件

- [BRPA 主源码](../BRPA.py) —— 注释清晰的可检索版本。
- [原始上传源码](../reference/BRPA_ablation_original.py) —— 字节级原样保存。
- [中英文项目首页](../README_zh-CN.md) —— 安装、调用、消融与文件结构。
- [GitHub 发布说明](GITHUB_PUBLISH_zh-CN.md) —— 新建仓库、设置主题标签、可被搜索。
