# RmGPT: Foundation Model for Rotating Machinery Fault Diagnosis and Prognosis

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-orange.svg)](https://github.com/yourusername/RmGPT)
[![Data](https://img.shields.io/badge/Dataset-Coming%20Soon-blue.svg)](https://github.com/yourusername/RmGPT)

## 📚 Overview

RmGPT is a unified foundation model designed for fault diagnosis and prognosis tasks in rotating machinery. Unlike traditional methods that require separate models for different equipment types, RmGPT uses a single set of parameters to handle diverse datasets with varying signal characteristics, fault modes, and operating conditions.

**Key Features:**
- 🔧 **Unified Framework**: Single model handles both diagnosis and prognosis tasks across different equipment
- 🚀 **Foundation Model**: Pre-trained on diverse unlabeled data for robust representation learning
- 🎯 **Few-shot Learning**: Achieves 82% accuracy in 16-class one-shot scenarios
- ⚡ **Efficient Design**: Patch-based tokenization and channel-time attention for computational efficiency
- 🎪 **Superior Performance**: Outperforms state-of-the-art algorithms across multiple datasets

## 🏗️ Architecture

RmGPT introduces a novel generative token-based framework with four key components:

1. **Signal Tokens**: Capture essential trends and patterns in input signals
2. **Prompt Tokens**: Enable task-specific adaptation and context awareness
3. **Time-Frequency Task Tokens**: Encapsulate health semantics for diagnosis/prognosis
4. **Fault Tokens**: Represent prototype characteristics of different fault modes

### Model Architecture Highlights

- **Patch-Based Tokenizer**: Segments time series into manageable patches for efficient processing
- **Channel-Time Attention**: Two-stage attention mechanism for multi-channel signal processing
- **Generative Pre-training**: Next signal token prediction for robust feature learning
- **Prompt Learning**: Parameter-efficient adaptation requiring <5% of total parameters

## 📊 Performance

### Main Results

| Task | Dataset | RmGPT-Prompt | RmGPT-Finetune | Best Baseline |
|------|---------|--------------|----------------|---------------|
| Diagnosis | CWRU | 99.92% | **100.00%** | 100.00% |
| Diagnosis | QPZZ | **100.00%** | **100.00%** | 100.00% |
| Diagnosis | SLIET | 98.05% | **99.02%** | 98.33% |
| Diagnosis | SMU | 99.11% | **100.00%** | 98.02% |
| Prognosis | XJTU | 0.150 MAE | **0.136 MAE** | 0.158 MAE |

### Few-shot Learning Performance (CWRU Dataset)

| Training Samples | RmGPT-Prompt | Best Baseline |
|------------------|--------------|---------------|
| 1-shot (16 samples) | **82.46%** | 21.47% |
| 4-shot (64 samples) | **98.53%** | 55.36% |
| 8-shot (128 samples) | **99.27%** | 95.08% |






## 📁 Dataset Support

RmGPT has been validated on multiple benchmark datasets:

- **CWRU**: Case Western Reserve University bearing dataset (16 classes, 1-3 channels)
- **SLIET**: Sant Longowal Institute bearing dataset (13 classes, 3 channels)  
- **QPZZ-II**: Gear fault dataset (5 classes, 9 channels)
- **SMU**: Gear dataset from Southern Methodist University (3 classes, 1 channel)
- **XJTU**: Bearing prognosis dataset for RUL prediction (4 classes, 2 channels)

> **📦 Processed Datasets Coming Soon**: We are preparing standardized, pre-processed versions of all benchmark datasets used in our experiments. These will be made available for easy reproduction of our results.

## 🔧 Configuration

## 📈 Evaluation

### Running Experiments

```bash
# Run diagnosis experiments
python experiments/diagnosis.py --dataset CWRU --config config/rmgpt_config.yaml

# Run prognosis experiments  
python experiments/prognosis.py --dataset XJTU --config config/rmgpt_config.yaml

# Few-shot learning evaluation
python experiments/few_shot.py --dataset CWRU --shots 1 4 8 16
```







## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.



## 📄 Citation

If you use RmGPT in your research, please cite our paper:

```bibtex
@article{wang2024rmgpt,
  title={RmGPT: A Foundation Model with Generative Pre-trained Transformer for Fault Diagnosis and Prognosis in Rotating Machinery},
  author={Wang, Yilin and Yu, Yifei and Sun, Kong and Lei, Peixuan and Zhang, Yuxuan and Zio, Enrico and Xia, Aiguo and Li, Yuanxiang},
  journal={IEEE Transactions on Industrial Electronics},
  year={2024},
  publisher={IEEE}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This work was partially supported by Grant of National Natural Science Foundation (Grant No. 62371297)
- Fund of Shanghai Engineering Research Center of Civil Aircraft Health Monitoring (Grant No. GCZX202204)
- iRel40 European co-funded innovation project (Grant No. 876659)

## 📞 Contact

<div align="center">

### Research Team

<table>
<tr>
<td align="center" width="200">
<img src="./pictures/sjtu.png" width="100"><br>
<b>Shanghai Jiao Tong University</b><br>
Yuanxiang Li (Corresponding)<br>
<i>yuanxli@sjtu.edu.cn</i><br>
Yilin Wang, Yifei Yu, Kong Sun<br>
Peixuan Lei, Yuxuan Zhang
</td>
<td align="center" width="200">
<img src="./pictures/PSL.png" width="100"><br>
<b>MINES Paris-PSL</b><br>
Enrico Zio<br>
<i>enrico.zio@polimi.it</i>
</td>
<td align="center" width="200">
<img src="./pictures/pm.png" width="100"><br>
<b>Politecnico di Milano</b><br>
Enrico Zio<br>
<i>enrico.zio@polimi.it</i>


<td align="center" width="200">
<img src="./pictures/sii.png" width="100"><br>
<b>Shanghai Innovation Institute</b><br>
Yilin Wang<br>
<i>pandalin@sjtu.edu.cn</i>
</td>
<td align="center" width="200">
</td>
</tr>
</table>

</div>

## 🔗 Links

- 📄 [Paper](link-to-paper)
- 📊 [Datasets](link-to-datasets) 
- 📚 [Documentation](link-to-docs)
- 🐛 [Issues](https://github.com/yourusername/RmGPT/issues)
- 💾 [Code Repository](https://github.com/yourusername/RmGPT) - **Coming Soon**

---

<div align="center">

**📢 Status Updates**

🚧 **Code Release**: Full implementation coming soon  
📦 **Dataset Release**: Processed datasets coming soon  
📖 **Documentation**: Comprehensive docs in preparation

**⭐ Star this repository to get notified when the code is released!**

</div>

---

**Note**: This is a research project. For production use, please thoroughly validate the model on your specific application and datasets.