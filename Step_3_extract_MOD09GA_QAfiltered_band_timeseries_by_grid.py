import os
import rasterio
import pandas as pd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
from collections import defaultdict

# === 用户设定路径 ===
grid_info_path = "/Users/athena/Desktop/MODIS_Script/Data_output/grid_info.xlsx"
image_folder = "/Volumes/Backup Plus/MODIS_MOD09GA_processed/MODIS_MOD09GA_clipped_2022"
output_folder = "/Users/athena/Desktop/MODIS_Script/Data_output"
year = "2022"

# === 读取网格信息 ===
df_grid = pd.read_excel(grid_info_path)
grid_ids = df_grid["Grid_ID"].tolist()
points = [Point(xy) for xy in zip(df_grid["Grid_Longitude"], df_grid["Grid_Latitude"])]

# === 初始化输出结构 ===
band_dict = {f"b0{i}": pd.DataFrame({"Grid_ID": grid_ids}) for i in range(1, 8)}

# === 扫描文件并整理为日期结构 ===
all_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".tif")])
doy_dict = defaultdict(lambda: {"bands": {}, "qc": None})

for fname in all_files:
    parts = fname.split("_")
    doy_part = [p for p in parts if p.startswith("doy")]
    if not doy_part:
        continue
    doy_part = [p for p in parts if "doy" in p]
    if not doy_part:
        continue
    doy_raw = doy_part[0]
    doy_number = ''.join(filter(str.isdigit, doy_raw))[-3:]
    full_date = f"{year}{doy_number}"
    if "_b0" in fname:
        band_part = [p for p in parts if p.startswith("b0")]
        if band_part:
            band = band_part[0]
            doy_dict[full_date]["bands"][band] = fname
    elif "QC_500m" in fname:
        doy_dict[full_date]["qc"] = fname

# === QA规则：MODLAND_QC 两位为 00 表示最高质量 ===
def is_good_quality(qc_val):
    return qc_val & 0b11 == 0

# === 主处理循环 ===
for full_date in tqdm(sorted(doy_dict.keys()), desc="提取每一日的波段值"):
    band_files = doy_dict[full_date]["bands"]
    qc_file = doy_dict[full_date]["qc"]

    qc_mask = None
    if qc_file:
        try:
            with rasterio.open(os.path.join(image_folder, qc_file)) as src_qc:
                qc_data = src_qc.read(1)
                qc_mask = np.vectorize(is_good_quality)(qc_data)
        except:
            print(f"⚠️ 读取 QC 文件失败：{qc_file}")
            qc_mask = None
    else:
        print(f"⚠️ 缺失 QC 文件：{full_date}，保留所有像元")

    for band in band_dict:
        if band in band_files:
            tif_path = os.path.join(image_folder, band_files[band])
            try:
                with rasterio.open(tif_path) as src:
                    values = []
                    for pt in points:
                        try:
                            row, col = src.index(pt.x, pt.y)
                            val = src.read(1)[row, col]
                            if qc_mask is not None and not qc_mask[row, col]:
                                val = np.nan
                            values.append(val)
                        except:
                            values.append(np.nan)
                    band_dict[band][full_date] = values
            except:
                print(f"❌ 无法读取波段文件：{tif_path}")
                band_dict[band][full_date] = [np.nan] * len(points)
        else:
            band_dict[band][full_date] = [np.nan] * len(points)

# === 保存为 CSV 文件 ===
for band, df in band_dict.items():
    out_path = os.path.join(output_folder, f"MOD09GA_QAfiltered_timeseries_by_grid_{band}_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ 已保存：{out_path}")