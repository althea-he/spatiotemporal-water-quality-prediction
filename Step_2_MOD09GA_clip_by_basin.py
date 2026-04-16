import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from tqdm import tqdm

# 输入路径和输出路径
input_folder = "/Volumes/Backup Plus/MODIS_MOD09GA_origin/2022"
output_folder = "/Volumes/Backup Plus/MODIS_MOD09GA_processed/MODIS_MOD09GA_clipped_2022"
shapefile_path = "/Users/athena/Desktop/本科毕设/Data_raw/流域范围/Basin.shp"

# 加载真实的流域范围
basin = gpd.read_file(shapefile_path)
basin = basin.to_crs(epsg=4326)  # 转为常见坐标系以防坐标不一致

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有tif图像
tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif") and not f.startswith("._")]

for tif_file in tqdm(tif_files, desc="裁剪MOD09GA图像"):
    input_path = os.path.join(input_folder, tif_file)
    output_path = os.path.join(output_folder, tif_file)
    
    try:
        with rasterio.open(input_path) as src:
            out_image, out_transform = mask(src, basin.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    except Exception as e:
        print(f"❌ 裁剪失败：{tif_file}，错误信息：{e}")