import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import os

# 设置输入输出路径
input_path = '/Users/athena/Desktop/本科毕设/6_建立水质预测模型_清水铺_时空/清水铺断面水文与水质与气象与波段数据.xlsx'
output_path = '/Users/athena/Desktop/本科毕设/6_建立水质预测模型_清水铺_时空/筛选模型输入变量_Qingshuipu.xlsx'

# 读取数据
df = pd.read_excel(input_path)

# 定义目标变量名称及对应sheet名称
targets = {
    '氨氮_变量重要性排序': '氨氮(mg/L)',
    '总磷_变量重要性排序': '总磷(mg/L)',
    '总氮_变量重要性排序': '总氮(mg/L)'
}

# 去除缺失值
df_clean = df.dropna()

# 初始化Excel写入器
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, target_col in targets.items():
        # 构建特征集（只剔除当前目标变量，保留其他变量）
        X_all = df_clean.drop(columns=[target_col])
        X_numeric = X_all.select_dtypes(include=[np.number])
        feature_names = X_numeric.columns

        # 目标变量
        y = df_clean[target_col]

        # 标准化数值特征
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=feature_names)

        # 计算 Pearson 相关性
        corr_series = X_numeric.corrwith(y)
        corr_df = pd.DataFrame({
            '变量名': corr_series.index,
            '相关系数': corr_series.values,
            '相关性绝对值': np.abs(corr_series.values)
        }).sort_values(by='相关性绝对值', ascending=False).reset_index(drop=True)

        # RFE + 随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=rf, n_features_to_select=1, step=1)
        rfe.fit(X_scaled, y)
        rfe_ranking = rfe.ranking_

        rfe_df = pd.DataFrame({
            '变量名': feature_names,
            'RFE排序': rfe_ranking
        })

        # 合并结果
        merged = pd.merge(corr_df, rfe_df, on='变量名')

        # 计算平均排名（用于综合排序，排名越小越重要）
        merged['相关性排名'] = merged['相关性绝对值'].rank(method='min', ascending=False)
        merged['平均排名'] = merged[['相关性排名', 'RFE排序']].mean(axis=1)
        merged = merged.sort_values(by='平均排名').reset_index(drop=True)

        # 写入每个目标变量的排序结果到 Excel
        merged.to_excel(writer, sheet_name=sheet_name, index=False)

        # 输出Top10变量名为一句话
        top10_names = merged.sort_values(by=["平均排名", "RFE排序"]).head(10)["变量名"].tolist()
        variable_name_clean = sheet_name.replace("_变量重要性排序", "")
        print(f"{variable_name_clean}的Top10变量为：" + "、".join(top10_names) + "。")

print("\n✅ 清水铺断面变量选择分析已完成，结果已保存为：", output_path)