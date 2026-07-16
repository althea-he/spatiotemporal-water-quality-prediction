# Spatiotemporal Water Quality Prediction

## Overview | 项目简介

This project develops a **data-driven watershed water quality prediction framework** based on multi-source environmental data.

本项目基于多源环境数据，构建了一个**流域水质时空预测模型**，融合遥感数据、气象数据与水质监测数据，实现关键水质指标的预测与预警。

---

## Key Features | 核心内容

- Multi-source data fusion（多源数据融合）
- Remote sensing preprocessing pipeline（遥感数据处理流程）
- Time series modeling with ANN（基于 ANN 的时间序列预测）
- Spatiotemporal modeling with STGCN（基于 STGCN 的时空建模）
- Water quality early warning system（水质预警机制）

---

## Methods | 方法

### 1. Data Processing Pipeline

The project includes a full preprocessing pipeline for MODIS remote sensing data:

- Step 1: Grid & subbasin mapping  
- Step 2: Basin clipping & validation  
- Step 3: QA filtering & time series extraction  
- Step 4: Missing value interpolation  
- Step 5: Multi-band feature construction  

👉 实现了完整的遥感数据处理流程（从裁剪到时间序列构建）

---

### 2. Modeling

#### ANN Models
Used for time-series prediction of:

- Ammonia Nitrogen (NH₄⁺)
- Total Nitrogen (TN)
- Total Phosphorus (TP)
- Permanganate Index

👉 用于建模时间维度变化

---

#### STGCN Model
A spatiotemporal graph convolutional network is used to capture:

- Spatial relationships between subbasins  
- Temporal dynamics of water quality  

👉 同时建模“空间 + 时间”特征

---

## Project Structure | 项目结构

```
├── Data/                          # Data folder (not included)
├── Step_1_*.py                    # Remote sensing preprocessing
├── Step_2_*.py
├── Step_3_*.py
├── Step_4_*.py
├── Step_5_*.py
├── ANN_*.py                       # ANN models
├── STGCN_*.py                     # STGCN model
├── feature_selection.py           # Feature selection
├── methodology_pipeline.jpg       # Methodology diagram
└── data_driven_water_quality_prediction_thesis.pdf
```
## Results | 结果
	•	Both ANN and STGCN models achieved strong predictive performance
	•	STGCN outperforms ANN in capturing spatial dependencies
	•	Significant improvements in RMSE, MAE, and MAPE for key indicators

👉 STGCN 在复杂流域场景下表现更优

---

## Data | 数据说明

Due to file size and data source restrictions, the raw dataset is not included in this repository.

由于数据体量较大及数据来源限制，原始数据未包含在仓库中。

Please refer to:
```
Data/README.md
```

---

## How to Run | 使用方式

```
# Step 1: Data preprocessing
Run Step_1 → Step_5 scripts

# Step 2: Feature engineering
python feature_selection.py

# Step 3: Train models
python ANN_*.py
python STGCN_*.py
```

---

## Applications | 应用场景
	•	Watershed management
	•	Environmental monitoring
	•	Water quality early warning
	•	Sustainable water resource planning


