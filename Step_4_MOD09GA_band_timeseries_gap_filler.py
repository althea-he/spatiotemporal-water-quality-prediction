import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

# === 输入输出路径设定 ===
input_folder = "/Users/athena/Desktop/MODIS_Script/Data_output"
output_folder = input_folder
grid_info_path = os.path.join(output_folder, "grid_info.xlsx")
year = "2022"
bands = [f"b0{i}" for i in range(1, 8)]
K = 4  # 最近邻个数

# === 读取网格坐标信息 ===
df_grid = pd.read_excel(grid_info_path)
grid_ids = df_grid["Grid_ID"].tolist()
coords = list(zip(df_grid["Grid_Longitude"], df_grid["Grid_Latitude"]))
tree = KDTree(coords)

# === 主循环：每个波段逐个处理 ===
for band in bands:
    input_file = os.path.join(input_folder, f"MOD09GA_QAfiltered_timeseries_by_grid_{band}_{year}.csv")
    output_file = os.path.join(output_folder, f"MOD09GA_QAfiltered_timeseries_by_grid_{band}_{year}_interpolated.csv")

    df = pd.read_csv(input_file, na_values=["", "NaN", -28672])
    df_interp = df.copy()
    ts_columns = df.columns[1:]
    df_interp[ts_columns] = df[ts_columns].astype(float)

    # === Step 1: 先做横向插值（行内）===
    df_interp[ts_columns] = df_interp[ts_columns].interpolate(axis=1, limit_direction='both')

    # === Step 2: 查找仍然整行缺失的网格 ===
    fully_nan_rows = df_interp[ts_columns].isna().all(axis=1)
    if fully_nan_rows.any():
        print(f"⚠️ {band}: {fully_nan_rows.sum()} 行完全缺失，尝试使用空间邻近网格均值填补...")

        # 建立 Grid_ID 到 DataFrame 行号映射
        id_to_index = {gid: idx for idx, gid in enumerate(grid_ids)}

        for row_idx in df_interp.index[fully_nan_rows]:
            this_coord = coords[row_idx]
            dist, neighbor_idxs = tree.query(this_coord, k=K + 1)  # 包含自己
            neighbor_idxs = [i for i in neighbor_idxs if i != row_idx]  # 排除自己

            for col in ts_columns:
                # 从邻居中收集非空值
                values = []
                for n_idx in neighbor_idxs:
                    val = df_interp.at[n_idx, col]
                    if not pd.isna(val):
                        values.append(val)
                if values:
                    df_interp.at[row_idx, col] = np.mean(values)

    # === Step 3: 最终缺失值统计 ===
    remaining_nans = df_interp[ts_columns].isna().sum().sum()
    if remaining_nans == 0:
        print(f"✅ 补全完成：{output_file}（无剩余缺失值）")
    else:
        print(f"⚠️ 补全完成：{output_file}，仍剩余 {int(remaining_nans)} 个 NaN")

    # === Step 4: 保存结果 ===
    df_interp.to_csv(output_file, index=False)