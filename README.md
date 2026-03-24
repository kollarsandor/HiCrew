

# 🚀 HiCrew

**[HiCrew: Hierarchical Reasoning for Long-Form
Video Understanding via Question-Aware
Multi-Agent Collaboration]**

[![Conference](https://img.shields.io/badge/ICME-2026-blue.svg)](https://www.icme2026.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<br>

<p align="center" style="font-size: 20px; font-weight: bold; color: #2e86ab;">
  🎉 Our work has been accepted by ICME 2026! 🎉
</p>


</div>

<hr>


## 📖 Overview

The following image illustrates the case study and the overall method of our **HiCrew** approach:

<div align="center">
  <img src="assets/case.png" width="80%" alt="HiCrew Method Overview" />
</div>

> **Note:** [Qualitative analysis on a Causal reasoning question from NExT-QA"]


---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/HiCrew.git
cd HiCrew
````

**2. Set up the environment**

```bash
conda create -n hicrew python=3.11 -y
conda activate hicrew
```

**3. Install dependencies**

```bash
pip install crewai
pip install openai                       
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas                      
pip install transformers==4.28.1         
pip install accelerate  
pip install opencv-python      
pip install numba               
pip install scikit-learn  
```

**4. Update Kmeans-pytorch**

```bash
git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch
```

Please replace the init file in "kmeans_pytorch" folder with the file we provide in "./HybridTree/kmeans_pytorch" folder (this repo). And run the following command. 

```bash
pip install --editable .
```

**4. Download dataset**


-----

## 📊 Performance

Our approach has been extensively evaluated. Here are our experimental results compared with other models:

<div align="center">
  <img src="assets/performance.png" width="60%" alt="HiCrew Method Overview" />
</div>

-----

## 📝 Citation

If you find our work or this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{hicrew2026,
  title     = {[HiCrew: Hierarchical Reasoning for Long-Form Video Understanding via Question-Aware Multi-Agent Collaboration]},
  author    = {[Yuehan Zhu, Jingqi Zhao and Jiawen Zhao]},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2026}
}
```

-----

## 📄 License

This project is licensed under the MIT License 
