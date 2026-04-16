import os
import geopandas as gpd
from shapely.geometry import box, Point
import pandas as pd

# ✅ 输入路径
basin_shp = "/Users/athena/Desktop/本科毕设/Data_raw/流域范围/Basin.shp"
subbasin_shp = "/Users/athena/Desktop/本科毕设/Data_raw/Subbasin/Export_Output.shp"
output_csv = "/Users/athena/Desktop/MODIS_Script/grid_info.csv"

# ✅ 加载数据
basin = gpd.read_file(basin_shp).to_crs(epsg=4326)
subbasin = gpd.read_file(subbasin_shp).to_crs(epsg=4326)

# ✅ 设置网格大小（500m ≈ 0.0045°）
grid_size_deg = 0.0045
minx, miny, maxx, maxy = basin.total_bounds

grid_polygons = []
grid_centers = []
grid_ids = []
current_id = 1

y = miny
while y < maxy:
    x = minx
    while x < maxx:
        cell = box(x, y, x + grid_size_deg, y + grid_size_deg)
        center = Point(x + grid_size_deg / 2, y + grid_size_deg / 2)
        if basin.contains(center).any():
            grid_polygons.append(cell)
            grid_centers.append(center)
            grid_ids.append(current_id)
            current_id += 1
        x += grid_size_deg
    y += grid_size_deg

grid_gdf = gpd.GeoDataFrame({
    "Grid_ID": grid_ids,
    "geometry": grid_centers
}, crs="EPSG:4326")

# ✅ 匹配子流域（shapefile 中字段为 'Subbasin'，输出字段重命名为 'Subbasin_ID'）
grid_with_subbasin = gpd.sjoin(grid_gdf, subbasin[["Subbasin", "geometry"]], how="left", predicate="within")
grid_with_subbasin = grid_with_subbasin.rename(columns={"Subbasin": "Subbasin_ID"})
grid_with_subbasin = grid_with_subbasin.drop(columns=["geometry", "index_right"])
grid_with_subbasin["Subbasin_ID"] = grid_with_subbasin["Subbasin_ID"].fillna(0).astype(int)

# ✅ 添加中心点坐标
grid_with_subbasin["Lon"] = [pt.x for pt in grid_centers]
grid_with_subbasin["Lat"] = [pt.y for pt in grid_centers]

# ✅ 导出 CSV
grid_with_subbasin.to_csv(output_csv, index=False)
print("✅ 网格信息已保存到：", output_csv)