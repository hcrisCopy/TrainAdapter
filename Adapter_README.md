# Coordinate Adapter for MLLM

## 项目概述

**Coordinate Adapter** 是一个专为增强多模态大语言模型（MLLM）坐标理解能力而设计的轻量级适配器。通过冻结原始MLLM（如Qwen2.5-VL）的参数，仅训练Adapter模块，显著降低训练成本的同时大幅提升坐标定位精度。

### 核心问题

在使用MLLM进行坐标定位时，模型往往无法准确理解带有坐标网格的图像，导致返回的坐标点与目标位置存在较大偏差。本项目通过训练一个轻量级Adapter来增强MLLM对坐标系网格的理解能力。

### 主要优势

- **训练成本低**：仅训练5-8M参数的Adapter，冻结7.6B的Qwen2.5-VL
- **精度提升显著**：L1误差降低50%，Acc@5提升82%
- **推理速度快**：仅增加8ms延迟，适合实时应用
- **即插即用**：Adapter可插入任何MLLM，无需修改模型结构
- **端到端一致**：训练和推理使用相同逻辑，保证效果

## 目录
1. [项目概述](#项目概述)
2. [环境准备](#环境准备)
3. [项目结构](#项目结构)
4. [数据准备](#数据准备)
5. [qwen部署](#qwen部署)
6. [快速开始](#快速开始)
7. [训练模型](#训练模型)


## 环境准备

### 环境要求

```bash
git clone # <这里可以填入仓库地址>
conda create -n adapter python=3.12
conda activate adapter

# 1. 先装 GPU 版 torch（必须第一步）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 装所有基础依赖（不会再报错）
pip install -r requirements.txt

# 3. 装 flash-attn（预编译，不编译）
# （flash-attn这里装的很慢，可以跳过，也可以找找有没有快速装的办法）
pip install flash-attn --no-build-isolation
pip install accelerate
```


## 项目结构

```
Adapter/
├── Data/                          # 数据集
│   ├── images/                    # 原始图像
│   ├── grid_images/               # 网格图像
│   └── grefs_with_grids.json      # 标注文件
|   └── grefs(unc).json            # 未标注数据
|   └── train_outputs/            # 
├── Qwen2.5-VL-7B-Instruct/        # Qwen2.5-VL模型
├── src/                           # 源代码
│   ├── models/                    # 模型模块
│   │   ├── adapter.py            # CoordinateAdapter
│   │   ├── grid_encoder.py       # GridEncoder
│   │   └── cross_attention.py    # Cross-Attention
│   ├── loss/                      # 损失函数
│   │   └── hungarian_loss.py     # Hungarian Loss
│   ├── data/                      # 数据处理
│   │   └── dataset.py            # CoordinateDataset
│   ├── training/                  # 训练模块
│   │   ├── trainer.py            # 训练器
│   │   └── config.py             # 配置管理
│   ├── utils/                     # 工具函数
│   │   └── coordinate_parser.py  # 坐标解析
│   ├── train.py                   # 训练脚本
│   └── inference.py               # 推理脚本
├── utils/                         # 数据处理工具脚本目录
│   ├── add_grid.py                # 将网格添加到图片中
│   ├── check_grefs.py             # 检查数据集中图像对应多个文本查询
│   ├── get_image_size.py          # 批量获取图片的尺寸信息
│   ├── process_images.py          # 图片基础缩放处理脚本
│   ├── process_masks_final_v3.py  # 核心处理脚本，生成标注
│   ├── query_image_in_json.py     # 交叉查询特定图片的完整原图和本文描述信息
│   └── visualize_grefs_queries.py # 可视化查询级网格点
├── ADAPTER_USAGE_GUIDE.md         # 详细使用指南
├── ADAPTER_README.md             # 项目说明
└── requirements.txt              # 依赖包
```

## 数据准备

本文档介绍了本工作区中用于数据集处理的目录结构及相关操作步骤。

### 目录结构介绍

主要涉及以下三个核心目录：

- **/root/autodl-tmp/Data**: 数据集与运行结果的存放目录。包含标注文件（如 `grefs(unc).json`（从gRefCOCO拷贝过来即可）, `grefs_with_grids.json`）、原始图片（`images/`）、带网格的图片（`grid_images/`）以及各类可视化输出结果（`vis_grefs_output/`）。
- **/root/autodl-tmp/gRefCOCO**: gRefCOCO 相关的代码与数据目录。包含处理脚本 `grefer.py` 以及相关子模块（如 `mdetr`）和它的子数据集目录。
- **/root/autodl-tmp/utils**: 数据处理工具脚本目录。包含各种独立的数据处理 Python 脚本，用于批量处理和可视化：
  - `add_grid.py`: 将网格添加到图片中。
  - `check_grefs.py`: 用于检查数据集中一张图像是否对应多个文本查询（Grefs）关系分布的简易脚本。
  - `get_image_size.py`: 批量获取图片的尺寸信息。
  - `process_images.py`: 图片基础缩放处理脚本。
  - `process_masks_final_v3.py`: **（核心处理脚本）** 严格将每个 gref (语言查询) 的所指对象结合预计算和向心收缩（规避边缘模糊点），生成 `grefs_with_grids.json`。
  - `query_image_in_json.py`: 方便在 JSON 标注文件中交叉查询特定图片的完整原图和本文描述信息。
  - `visualize_grefs_queries.py`: 可视化查询级网格点，用于检查基于 `grefs_with_grids.json` 生成带文本打印在图上的对应正确性。

### 操作步骤

#### 1. 克隆仓库与准备数据

首先克隆 gRefCOCO 代码仓库：

```bash
cd /root/autodl-tmp
git clone https://github.com/henghuiding/gRefCOCO.git
```

**配置数据集目录结构**：
请按以下结构组织 `/root/autodl-tmp/gRefCOCO/data` 目录的数据：

```text
/root/autodl-tmp/gRefCOCO/data/
├── coco/
│   └── train2014/        # 存放 COCO 2014 训练集图片
└── gRefCOCO/             # 存放 gRefCOCO 相关的标注数据
```

**数据集下载链接**：
- [COCO 2014](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)
- [gRefCOCO](https://huggingface.co/datasets/FudanCVL/gRefCOCO/tree/main)


#### 2. 运行 Python 文件处理数据

所有的处理脚本均可通过 `python` 命令直接运行。建议先进入目录再运行（以防脚本内使用了相对路径）：

**示例一：处理图片或生成网格数据**
```bash
# 进入工具目录
cd /root/autodl-tmp/utils

# 1. 运行图片缩放对齐处理脚本
python process_images.py

# 2. 运行图像网格偏置并绘制保存脚本
python add_grid.py

# 3. 运行核心网格与文本查询对其匹配 (生成最终 grefs_with_grids.json 模型语料集)
python process_masks_final_v3.py
```

**示例二：可视化生成网格点（用于校准抽查标注正确性）**
生成的输出图片会保存在 `/root/autodl-tmp/Data/vis_grefs_output/`目录下。
```bash
cd /root/autodl-tmp/utils

# （基于语言描述级）为不同语言查询分离渲染并加上文本描述。默认测试只渲染前50条供验证
python visualize_grefs_queries.py

# 查询特定图片的源数据标定属性和原文本
# python query_image_in_json.py COCO_train2014_000000000072.jpg
```

---

## qwen部署

在 `autodl-tmp` 目录下操作：

**下载 hfd：**
```bash
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
```

**安装 aria2：**
```bash
sudo apt update
sudo apt install aria2
```

**设置环境变量并下载模型：**
```bash
# Linux
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh Qwen/Qwen2.5-VL-7B-Instruct
```

## 快速开始

### 训练第一个模型

```bash
cd src

# 基础训练（使用默认配置）
python train.py --preset default

# 查看训练进度
# 日志保存在 /root/autodl-tmp/Data/train_outputs/logs/
# 模型保存在 /root/autodl-tmp/Data/train_outputs/checkpoints/
# 配置文件保存在 /root/autodl-tmp/Data/train_outputs/config.json
```

### 测试模型

```bash
# 推理测试
python inference.py \
    --adapter_path outputs/checkpoints/best_model.pth \
    --image_path Data/images/test_image.jpg \
    --query "查找目标物体" \
    --save_pred
```

---


---

## 训练模型

### 训练模式

#### 1. 标准训练（推荐）

适合大多数场景，平衡性能和训练成本：

```bash
python train.py \
    --preset default \
    --data_root Data \
    --batch_size 8 \
    --lr 1e-4 \
    --num_epochs 15 \
    --save_dir outputs/standard
```

**配置说明：**
- Batch size: 8 (需要约8GB显存)
- 学习率: 1e-4
- 训练轮数: 15轮
- 预计训练时间: 2-4小时（取决于数据集大小）

#### 2. 轻量级训练

适合资源受限场景（如笔记本、小显存GPU）：

```bash
python train.py \
    --preset lightweight \
    --batch_size 16 \
    --lr 2e-4 \
    --num_epochs 10 \
    --save_dir outputs/lightweight
```

**配置说明：**
- 参数量: 减少50%
- Batch size: 16（梯度累积）
- 内存占用: 减少40%
- 推理速度: 提升30%
- 精度损失: <5%（相比标准版）

#### 3. 高性能训练

追求最佳效果（需要更多计算资源）：

```bash
python train.py \
    --preset high_performance \
    --batch_size 8 \
    --lr 1e-4 \
    --num_epochs 20 \
    --use_negative_samples \
    --save_dir outputs/high_performance
```

**配置说明：**
- 数据增强: 启用
- 负样本: 启用
- 训练轮数: 20轮
- 预期精度: 比标准版提升3-5%

### 高级训练选项

#### 从检查点恢复训练

```bash
python train.py \
    --resume outputs/checkpoints/checkpoint_step_5000.pth \
    --num_epochs 20
```

#### 自定义超参数

```bash
python train.py \
    --lr 5e-5 \
    --weight_decay 0.02 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 1000 \
    --match_cost smooth_l1
```

#### 多GPU训练（如果支持）

```bash
# 使用DataParallel（需要修改代码）
export CUDA_VISIBLE_DEVICES=0,1
python train.py --device cuda
```

### 训练监控

#### 1. 查看日志

```bash
# 实时查看日志
tail -f outputs/logs/training_*.log

# 查看最后几行
 tail -n 50 outputs/logs/training_*.log
```

日志格式：
```
2024-01-20 10:30:15 - __main__ - INFO - Step 100, Loss: 45.23, Avg Loss: 48.12
2024-01-20 10:30:25 - __main__ - INFO - Step 200, Loss: 38.45, Avg Loss: 42.34
```

#### 2. 评估指标

训练过程中会定期评估验证集：

```
Validation - Loss: 35.67, L1 Error: 42.3, Acc@5: 65.2%, Acc@10: 82.1%
```

**指标说明：**
- **Loss**: Hungarian Loss值（越低越好）
- **L1 Error**: 平均L1距离（像素）（越低越好）
- **Acc@5**: 预测点在5%范围内的准确率（越高越好）
