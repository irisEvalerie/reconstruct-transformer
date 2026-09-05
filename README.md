# Reconstruct Transformer

从基础张量运算出发，复现 *Attention Is All You Need* 中的经典
Encoder–Decoder Transformer。

## 项目状态

当前源码最初由跟随 B 站“手搓 Transformer”教程编写而成，保留为可运行的
教程基线。项目现已进入**系统验证与研究化重构阶段**：后续将逐个拆分组件、
独立推导公式、补充数值与梯度测试，并通过合成 copy task 和小型翻译实验验证
完整模型。

教程为学习路径和初始代码结构提供了参考；本仓库不会把教程内容包装成原创。
正式教程链接将在文档整理阶段补充，后续提交将记录独立验证和实验过程。

## 当前实现

- Token Embedding 与正弦位置编码
- Scaled Dot-Product Attention 与 Multi-Head Attention
- 手写 Layer Normalization 与 Position-wise Feed-Forward Network
- Padding Mask 与 Causal Mask
- Encoder、Decoder 与完整 Encoder–Decoder Transformer

> 当前完整模型属于“待系统验证的教程骨架”，不能仅凭一次前向传播认定实现
> 正确。组件测试与训练实验将在后续里程碑中逐步加入。

## 安装

项目要求 Python 3.12。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

需要 NVIDIA GPU 时，请先根据
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装匹配的稳定版
CUDA wheel，再执行 editable install。

## 验证

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

## 项目结构

```text
reconstructTransformer/
├─ configs/                       # 模型、训练和实验配置
├─ data/{raw,processed}/           # 本地数据，不提交大文件
├─ docs/lessons/                  # 推导、决策和实验笔记
├─ notebooks/                     # 公式验证与可视化
├─ scripts/                       # 环境、训练、推理和评估入口
├─ src/reconstruct_transformer/   # 模型源码
├─ tests/{unit,integration}/       # 组件与端到端测试
├─ checkpoints/                   # 本地权重，不提交
└─ outputs/                       # 本地实验产物，不提交
```

## 复现路线

1. 保存并验证教程基线
2. 独立验证 Scaled Dot-Product Attention
3. 验证 Embedding、LayerNorm、FFN 和 Masks
4. 重构并验证 Multi-Head Attention
5. 验证 Encoder、Decoder 与完整模型
6. 在合成 copy task 上验证训练与解码
7. 完成小型机器翻译实验、可视化与技术报告

## Reference

Vaswani, A. et al. (2017). [Attention Is All You
Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information
Processing Systems 30*.
