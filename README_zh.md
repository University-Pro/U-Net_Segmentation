# U-Net 分割项目

## 概述
欢迎来到 **U-Net 分割项目**！这是一个强大且灵活的医学图像分割工具。本项目提供了 **U-Net** 的实现，支持在 **Synapse** 和 **ACDC** 数据集上的分割任务。项目设计注重简洁和模块化，用户可以通过 **Dice 相似系数 (DSC)** 和 **Hausdorff 距离 (HD95)** 等指标对模型进行训练、测试和评估。

如果您想定制网络架构，可以将提供的 `network.py` 文件替换为您自己的实现。

---

## 主要特点
- **模块化设计**：轻松替换网络架构，支持自定义。
- **全面评估指标**：通过 DSC 和 HD95 评估分割质量。
- **数据集兼容性**：开箱即用地支持 Synapse 和 ACDC 数据集。
- **简化的安装配置**：使用 Anaconda 或 Miniconda 轻松管理环境。

---

## 安装与配置

### 先决条件
建议使用 **Anaconda** 或 **Miniconda** 来管理依赖和 Python 版本。在开始前，请先安装 Anaconda 或 Miniconda。

### 步骤 1：创建虚拟环境
运行以下命令创建虚拟环境：

```bash
conda create -n torch-env python=3.12
```

### 步骤 2：激活环境
通过以下命令激活虚拟环境：

```bash
conda activate torch-env
```

### 步骤 3：安装必要的依赖
使用 `requirements.txt` 文件安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

> **提示**：如果访问官方 PyPI 仓库速度较慢或受限，可以使用镜像（例如 [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)）。

---

## 使用方法

### 训练模型
要在您的数据集上训练 U-Net 模型：
1. 将数据集放置在配置文件指定的目录中。
2. 运行训练脚本：

```bash
python train.py --config configs/config.yaml
```

### 测试模型
训练完成后，您可以在测试集上评估模型：

```bash
python test.py --model checkpoints/best_model.pth --config configs/config.yaml
```

### 自定义网络
将 `network.py` 文件替换为您自己的网络实现。请确保新文件遵循训练和测试脚本中使用的输入输出规范。

---

## 数据集准备
确保您的数据集结构如下：
```
datasets/
├── Synapse/
│   ├── train/
│   ├── test/
├── ACDC/
│   ├── train/
│   ├── test/
```
您可以根据需要修改配置文件 (`configs/config.yaml`) 中的路径。

---

## 评估指标
- **Dice 相似系数 (DSC)**：衡量预测分割与真实标签之间的重叠。
- **Hausdorff 距离 (HD95)**：评估分割边界的空间精度。

---

## 常见问题
- **安装速度慢**：使用 PyPI 镜像，例如 TUNA。
- **CUDA 问题**：确保您的 GPU 驱动程序和 CUDA 工具包正确安装，并与 PyTorch 版本匹配。
- **内存不足 (OOM)**：在配置文件中减小批量大小。

---

## 贡献
欢迎贡献代码！您可以提交问题、功能请求或拉取请求。

---

## 许可证
本项目使用 [MIT License](LICENSE) 授权。

---

## 致谢
感谢开源社区提供的数据集和对本项目的启发！

---

## 联系方式
如有问题或需要支持，请联系项目维护者：
- **邮箱**: support@example.com
- **GitHub Issues**: [提交问题](https://github.com/your-repo/issues)

祝您分割愉快！🚀