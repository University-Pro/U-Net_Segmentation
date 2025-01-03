# U-Net Segmentation Project

## Overview
Welcome to the **U-Net Segmentation Project**, a powerful and flexible tool for medical image segmentation. This repository provides an implementation of **U-Net** for segmentation tasks on datasets like **Synapse** and **ACDC**. The project is designed for simplicity and modularity, allowing users to train, test, and evaluate models using metrics such as **Dice Similarity Coefficient (DSC)** and **Hausdorff Distance (HD95)**.

If you'd like to customize the network architecture, you can replace the provided `network.py` file with your own implementation.

---

## Key Features
- **Modular Design**: Easily swap out the network with your custom architecture.
- **Comprehensive Metrics**: Evaluate segmentation quality using DSC and HD95.
- **Dataset Compatibility**: Supports Synapse and ACDC datasets out of the box.
- **Streamlined Setup**: Simplified environment management using Anaconda or Miniconda.

---

## Installation and Setup

### Prerequisites
It is recommended to use **Anaconda** or **Miniconda** for managing dependencies and Python versions. Install Anaconda or Miniconda before proceeding.

### Step 1: Create a Virtual Environment
Run the following command to create a virtual environment:

```bash
conda create -n torch-env python=3.12
```

### Step 2: Activate the Environment
Activate the virtual environment with:

```bash
conda activate torch-env
```

### Step 3: Install Required Packages
Install all necessary dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> **Tip**: If you experience slow or blocked access to the official PyPI repository, consider using a mirror (e.g., [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)).

---

## Usage

### Training the Model
To train the U-Net model on your dataset:
1. Place your dataset in the appropriate directory as specified in the configuration file.
2. Run the training script:

```bash
python train.py --config configs/config.yaml
```

### Testing the Model
After training, you can evaluate the model on the test set:

```bash
python test.py --model checkpoints/best_model.pth --config configs/config.yaml
```

### Customizing the Network
Replace the `network.py` file with your own network implementation. Ensure the new file adheres to the input-output specifications used in the training and testing scripts.

---

## Dataset Preparation
Ensure your datasets are structured as follows:
```
datasets/
├── Synapse/
│   ├── train/
│   ├── test/
├── ACDC/
│   ├── train/
│   ├── test/
```
You can modify the paths in the configuration files (`configs/config.yaml`) as needed.

---

## Metrics
- **Dice Similarity Coefficient (DSC)**: Measures the overlap between predicted and ground truth masks.
- **Hausdorff Distance (HD95)**: Evaluates the spatial accuracy of the segmentation boundaries.

---

## Troubleshooting
- **Slow Installation**: Use a PyPI mirror like TUNA.
- **CUDA Issues**: Ensure your GPU drivers and CUDA toolkit are correctly installed and match the PyTorch version.
- **Out of Memory (OOM) Errors**: Reduce the batch size in the configuration file.

---

## Contributions
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
Thanks to the open-source community for providing datasets and inspiring the development of this project.

---

## Contact
For questions or support, feel free to contact the project maintainer:
- **Email**: support@example.com
- **GitHub Issues**: [Open an issue](https://github.com/your-repo/issues) 

Happy Segmenting! 🚀