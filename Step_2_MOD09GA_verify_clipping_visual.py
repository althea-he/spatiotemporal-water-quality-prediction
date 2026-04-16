
import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import geopandas as gpd

# 文件路径（原图和裁剪图）
original_file = "/Volumes/Backup Plus/MODIS_MOD09GA_origin/2019/MOD09GA.061_sur_refl_b01_1_doy2019001_aid0001.tif"
clipped_file  = "/Volumes/Backup Plus/MODIS_MOD09GA_processed/MODIS_MOD09GA_clipped_2019/MOD09GA.061_sur_refl_b01_1_doy2019001_aid0001.tif"
shapefile_path = "/Users/athena/Desktop/本科毕设/Data_raw/流域范围/Basin.shp"

# 加载 Basin 边界
basin = gpd.read_file(shapefile_path)
basin = basin.to_crs("EPSG:4326")

# 加载图像
with rasterio.open(original_file) as src_orig:
    orig_img = src_orig.read(1)
    orig_bounds = src_orig.bounds

with rasterio.open(clipped_file) as src_clip:
    clip_img = src_clip.read(1)
    clip_bounds = src_clip.bounds

# 可视化对比（含 Basin 边界线）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
show(orig_img, ax=ax1, cmap='gray', title="原始图像（未经裁剪）")
basin.boundary.plot(ax=ax1, edgecolor='red', linewidth=1.5)

show(clip_img, ax=ax2, cmap='gray', title="裁剪后图像（Basin范围）")
basin.boundary.plot(ax=ax2, edgecolor='red', linewidth=1.5)

# 辅助信息输出
print("原图范围：", orig_bounds)
print("裁剪后图像范围：", clip_bounds)

plt.tight_layout()
plt.show()