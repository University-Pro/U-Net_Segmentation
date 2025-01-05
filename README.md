# 🌟 U-Net Segmentation Project

## 📝 Overview
Welcome to the **U-Net Segmentation Project**! 🚀 This project is a powerful and flexible tool for **medical image segmentation**, built with a robust implementation of **U-Net** for datasets like **Synapse** and **ACDC**. Designed for **simplicity** and **modularity**, it allows you to:
- Train, test, and evaluate models using **Dice Similarity Coefficient (DSC)** and **Hausdorff Distance (HD95)** metrics.
- Easily swap out the network architecture by replacing the `network.py` file with your custom implementation.

---

## ✨ Key Features
- 🔧 **Modular Design**: Easily customize the network with your own architecture.
- 📊 **Comprehensive Metrics**: Evaluate segmentation quality using **DSC** and **HD95**.
- 📂 **Dataset Compatibility**: Seamlessly supports **Synapse** and **ACDC** datasets.
- ⚡ **Streamlined Setup**: Simplified environment management using **Anaconda** or **Miniconda**.

---

## ⚙️ Installation and Setup

### 🛠 Prerequisites
We recommend using **Anaconda** or **Miniconda** to manage dependencies and Python versions. Install Anaconda or Miniconda before proceeding.

### 🪄 Step 1: Create a Virtual Environment
Run the following command to create a virtual environment:

```bash
conda create -n torch-env python=3.12
```

### 🚀 Step 2: Activate the Environment
Activate the virtual environment with:

```bash
conda activate torch-env
```

### 📦 Step 3: Install Required Packages
Install all necessary dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> 💡 **Tip**: If you experience slow or blocked access to the official PyPI repository, consider using a mirror like [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/).

---

## 🖥️ Usage

### 📚 Training the Model
To train the **U-Net** model on your dataset:
1. Place your dataset in the appropriate directory as specified in the configuration file.
2. Run the training script:

```bash
python train.py --config configs/config.yaml
```

### 🧪 Testing the Model
After training, evaluate the model on the test set with the following command:

```bash
python test.py --model checkpoints/best_model.pth --config configs/config.yaml
```

### 🔨 Customizing the Network
Replace the `network.py` file with your own network implementation. Ensure the new file adheres to the **input-output specifications** used in the training and testing scripts.

---

## 📂 Dataset Preparation
Organize your datasets as follows:
```
datasets/
├── Synapse/
│   ├── train/
│   ├── test/
├── ACDC/
│   ├── train/
│   ├── test/
```
You can modify the paths in the configuration files (`configs/config.yaml`) to match your setup.

---

## 📏 Metrics
- 🧮 **Dice Similarity Coefficient (DSC)**: Measures the overlap between predicted and ground truth masks.
- 📐 **Hausdorff Distance (HD95)**: Evaluates the spatial accuracy of the segmentation boundaries.

---

## 🛠 Troubleshooting
- 🐢 **Slow Installation**: Use a PyPI mirror like **TUNA**.
- ❌ **CUDA Issues**: Ensure GPU drivers and the CUDA toolkit are properly installed and compatible with your PyTorch version.
- ⚠️ **Out of Memory (OOM) Errors**: Reduce the batch size in the configuration file.

---

## 🤝 Contributions
We welcome contributions! 💡 Feel free to submit issues, feature requests, or pull requests. Let's make this project even better together!

---

## 📜 License
This project is licensed under the [MIT License](LICENSE). 📄

---

## 🌟 Acknowledgments
Thanks to the amazing **open-source community** for providing datasets and inspiring the development of this project! 🙌

---

## 📬 Contact
For questions or support, feel free to contact the project maintainer:
- 📧 **Email**: support@example.com
- 🐛 **GitHub Issues**: [Open an issue](https://github.com/your-repo/issues) 

Happy Segmenting! 🎉