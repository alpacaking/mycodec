# 语音编解码器实验项目

## 项目概述

本项目实现了基于 RVQGAN 范式的 Codec ，并进行了不同向量量化方式（RVQ和FSQ）以及不同参数配置的性能比较实验。项目使用LibriSpeech数据集训练和评估模型，通过PESQ和STOI等指标比较不同配置的语音重建质量。

## 代码框架
- Academic Codec 

## 项目结构

```
├── academicodec/                 # 核心编解码器实现
├── experiments/                  # 训练模型权重存储
├── evaluation_results/           # 评估结果存储，包含示例
├── analysis_results/             # 分析结果和图表
├── train.sh                      # 基准模型训练脚本
├── train_rqv4.sh                 # RVQ n_q_max=4 训练脚本
├── train_rqv12.sh                # RVQ n_q_max=12 训练脚本
├── train_bins512.sh              # RVQ bins=512 训练脚本
├── train_bins2048.sh             # RVQ bins=2048 训练脚本
├── train_fsq8655.sh              # FSQ [8,6,5,5] 训练脚本
├── test_all_models.sh            # 模型评估脚本
├── analyze_results.py            # 结果分析脚本
├── experiment.md                 # 实验报告
└── README.md                     # 项目说明
```

## 实验配置与训练脚本

### 1. RVQ不同码本层数实验

测试不同n_q_max值对性能的影响。

**训练脚本:**

- 基准模型 (n_q_max=8, bins=1024)
  ```bash
  ./train.sh
  ```

- n_q_max=4, bins=1024
  ```bash
  ./train_rqv4.sh
  ```

- n_q_max=12, bins=1024
  ```bash
  ./train_rqv12.sh
  ```

### 2. RVQ不同码本大小实验

测试不同bins值对性能的影响。

**训练脚本:**

- bins=512, n_q_max=8
  ```bash
  ./train_bins512.sh
  ```

- bins=2048, n_q_max=8
  ```bash
  ./train_bins2048.sh
  ```

### 3. FSQ实验

测试FSQ量化方式。

**训练脚本:**

- FSQ [8,6,5,5]
  ```bash
  ./train_fsq8655.sh
  ```

## 模型评估

使用test_all_models.sh脚本评估所有训练好的模型:

```bash
./test_all_models.sh
```

该脚本对每个RVQ模型在三个不同比特率(1.5, 3.0, 6.0 kbps)下进行评估，对FSQ模型在其固有比特率下进行评估。

## 结果分析

使用analyze_results.py脚本分析评估结果:

```bash
python analyze_results.py
```

该脚本会生成多种比较图表，包括：
- RVQ不同码本层数比较
- RVQ不同码本大小比较
- FSQ与RVQ比较
- 比特率与质量关系图

所有图表和结果摘要保存在analysis_results目录中。

## 实验结果

实验结果见experiment.md    

## 训练好的模型位置

所有训练好的模型保存在experiments目录下：

- 基准模型: experiments/librispeech_rvqgan_base/2025-06-13-03-56/latest.pth
- n_q_max=4: experiments/librispeech_rvqgan_rqv4/2025-06-13-03-58/latest.pth
- n_q_max=12: experiments/librispeech_rvqgan_rqv12/2025-06-13-03-58/latest.pth
- bins=512: experiments/librispeech_rvqgan_bins512/2025-06-13-03-56/latest.pth
- bins=2048: experiments/librispeech_rvqgan_bins2048/2025-06-13-03-57/latest.pth
- FSQ [8,6,5,5]: experiments/librispeech_rvqgan_fsq8655/2025-06-13-03-57/latest.pth

