# MOD09GA_b0x_2019001_2022365.py
# ====================================
# 合并2019–2022年各波段插值数据，填补缺失日期，确保时间序列完整

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === 参数设定 ===
data_folder = "/Users/athena/Desktop/MODIS_Script/Data_output"
output_folder = data_folder
years = ["2019", "2020", "2021", "2022"]
bands = [f"b0{i}" for i in range(1, 8)]

# === 构建完整的 DOY 日期列表 ===
def generate_all_doys(start_year, end_year):
    all_doys = []
    for year in range(start_year, end_year + 1):
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        delta = (end - start).days + 1
        for i in range(delta):
            day = start + timedelta(days=i)
            doy_str = f"{year}{day.timetuple().tm_yday:03d}"
            all_doys.append(doy_str)
    return all_doys

# === 获取完整 DOY 日期序列 ===
full_doy_list = generate_all_doys(2019, 2022)

# === 按波段循环处理 ===
for band in bands:
    print(f"\n🔄 处理波段：{band}")

    # === 读取四年数据 ===
    df_list = []
    for y in years:
        file_path = os.path.join(data_folder, f"MOD09GA_QAfiltered_timeseries_by_grid_{band}_{y}_interpolated.csv")
        df = pd.read_csv(file_path)
        df_list.append(df)

    # === 合并四年数据 ===
    df_merged = df_list[0].copy()
    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on="Grid_ID", how="outer")

    df_merged = df_merged.sort_values("Grid_ID").reset_index(drop=True)

    # === 检查和插入缺失日期列 ===
    existing_doys = [col for col in df_merged.columns if col != "Grid_ID"]
    missing_doys = [doy for doy in full_doy_list if doy not in existing_doys]

    if missing_doys:
        print(f"⚠️ 发现缺失日期列 {len(missing_doys)} 个，将插入空值：{missing_doys}")
        for doy in missing_doys:
            df_merged[doy] = np.nan

    # === 重新排序列（按 DOY 时间顺序）===
    sorted_doys = sorted([col for col in df_merged.columns if col != "Grid_ID"])
    df_merged = df_merged[["Grid_ID"] + sorted_doys]

    # === 将 DOY 列名转换为 YYYY-MM-DD 格式 ===
    def doy_to_date(doy_str):
        year = int(doy_str[:4])
        doy = int(doy_str[4:])
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return date.strftime("%Y-%m-%d")

    date_map = {doy: doy_to_date(doy) for doy in sorted_doys}
    df_merged.rename(columns=date_map, inplace=True)

    # === 插值填补缺失列数据 ===
    ts_cols = df_merged.columns[1:]
    df_merged[ts_cols] = df_merged[ts_cols].astype(float)

    # Step 1: 时间轴双向插值（linear）
    df_merged[ts_cols] = df_merged[ts_cols].interpolate(axis=1, limit_direction='both')

    # Step 2: 若仍有 NaN，使用前后值填补
    df_merged[ts_cols] = df_merged[ts_cols].fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)

    # 检查是否仍有缺失值
    remaining_na = df_merged[ts_cols].isna().sum().sum()
    if remaining_na == 0:
        print(f"✅ 补齐完成：{band}，共 {len(ts_cols)} 天，无缺失。")
    else:
        print(f"⚠️ 补齐后仍有 {int(remaining_na)} 个缺失值。")

    # === 保存最终输出文件 ===
    out_file = os.path.join(output_folder, f"MOD09GA_{band}_2019001_2022365.csv")
    df_merged.to_csv(out_file, index=False)
    print(f"📁 已保存文件：{out_file}")