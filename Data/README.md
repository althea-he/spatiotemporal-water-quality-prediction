# Data Folder Instructions

## Overview
This project uses multi-source environmental and remote sensing data for watershed water quality prediction.

Due to file size and data source restrictions, the raw dataset is not included in this repository.

---

## Data Structure

Please organize the data as follows:

```
Data/
├── DEM/
├── Landuse/
├── Soil/
├── Subbasin/
├── 河网/
├── 流域范围/
├── 气象_水文_水质_污水处理厂_断面_Data_raw/
├── 气象_水文_水质_污水处理厂_断面_Data_cleaned/
├── 遥感数据_MODIS021km/
└── 赤水河流域（云南段）水质监测断面经纬度.xlsx
```
---

## Data Sources

- MODIS Remote Sensing Data (MOD09GA / MOD021KM)  
  https://lpdaac.usgs.gov/

- DEM / Landuse / Soil data  
  Public geospatial datasets

- Hydrological and water quality data  
  Monitoring stations

---

## How to Use

1. Download required datasets
2. Place them into corresponding folders under `Data/`
3. Run preprocessing scripts:

Step_1 → Step_5

---

## Notes

This repository focuses on modeling and methodology.  
Data is excluded to keep the repo lightweight.
