import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, PReLU, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === 路径设置 ===
file_path = "/Users/athena/Desktop/本科毕设/4_建立水质预测模型_清水铺_ANN/变量选择结果_ANN输入数据整理.xlsx"
xls = pd.ExcelFile(file_path)
df = xls.parse(sheet_name="高锰酸盐_输入变量")  

# === 时间处理与特征工程 ===
df["日期"] = pd.to_datetime(df["日期"])
df.set_index("日期", inplace=True)
df["weekday"] = df.index.weekday
df["month"] = df.index.month
df["quarter"] = df.index.quarter
df["dayofyear"] = df.index.dayofyear

# Winsorize处理
def winsorize(series, lower_quantile=0.01, upper_quantile=0.99):
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return np.clip(series, lower, upper)

df = df.apply(lambda x: winsorize(x) if x.name not in ["weekday", "month", "quarter"] else x)

# 添加滞后特征
lags = [1, 2, 3, 7, 14, 30]
for lag in lags:
    df[f"高锰酸盐_lag{lag}"] = df["高锰酸盐指数(mg/L)"].shift(lag)

df["高锰酸盐_MA3"] = df["高锰酸盐指数(mg/L)"].rolling(3).mean()
df["高锰酸盐_MA7"] = df["高锰酸盐指数(mg/L)"].rolling(7).mean()
df["高锰酸盐_MA14"] = df["高锰酸盐指数(mg/L)"].rolling(14).mean()
df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
df = df.drop(columns=["dayofyear"]).dropna()

# === 归一化 ===
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# === 特征选择 ===
target_col = "高锰酸盐指数(mg/L)"
X = df_scaled.drop(columns=[target_col])
y = df_scaled[target_col]

# === 拆分数据集 ===
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_val, y_val = X.iloc[train_size:train_size + val_size], y.iloc[train_size:train_size + val_size]
X_test, y_test = X.iloc[train_size + val_size:], y.iloc[train_size + val_size:]

# === 构建模型 ===
def build_ann_model(input_dim, hidden_layers, dropout_rate):
    inp = Input(shape=(input_dim,))
    x = Dense(hidden_layers[0], kernel_regularizer=l2(0.001))(inp)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x_res = Dense(hidden_layers[1], kernel_regularizer=l2(0.001))(x)
    x_res = PReLU()(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = Dropout(dropout_rate)(x_res)

    x = Add()([x, x_res])

    x = Dense(hidden_layers[2], activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='log_cosh', metrics=['mae'])
    return model

# === 训练模型 ===
hidden_layers = (256, 256, 128)
dropout_rate = 0.1
model = build_ann_model(X_train.shape[1], hidden_layers, dropout_rate)

early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=5, min_lr=1e-5)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=300,
          batch_size=64,
          verbose=0,
          callbacks=[early_stopping, reduce_lr])

# === 预测 ===
y_pred_test = model.predict(X_test).flatten()

y_test_inv = y_test * (df[target_col].max() - df[target_col].min()) + df[target_col].min()
y_pred_test_inv = y_pred_test * (df[target_col].max() - df[target_col].min()) + df[target_col].min()

# === 预警 ===
threshold = 4
warning_mask = y_pred_test_inv > threshold  # 找到所有超标的预测点

# 提取超标日期和预测值
test_index = df.index[train_size + val_size:]  # 获取测试数据的日期索引
warning_dates = test_index[warning_mask]
warning_values = y_pred_test_inv[warning_mask]

# === 绘图 ===
plt.figure(figsize=(12, 6))

# 绘制实际和预测值
plt.plot(test_index, y_test_inv, label="Actual", color='blue', alpha=0.6)
plt.plot(test_index, y_pred_test_inv, label="Predicted", color='red', linestyle="dashed", alpha=0.8)

# 画出超标点
if len(warning_dates) > 0:  # 如果存在超标点
    plt.scatter(warning_dates, warning_values, color='red', label="⚠️ Exceeded", marker='o', s=50, edgecolors='black')

# 画出阈值线
plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1, label="Threshold (4 mg/L)")

plt.xlabel("Date")
plt.ylabel("Permanganate Index (mg/L)")
plt.title("Actual vs Predicted Permanganate Index with Warnings")

# 设置每月 1 号的日期标签，并确保最后一天也显示
xticks = pd.date_range(start=test_index.min(), end=test_index.max(), freq='MS').tolist()
if test_index.max() not in xticks:
    xticks.append(test_index.max())  # 添加最后一天

plt.xticks(xticks)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xlim(test_index.min(), test_index.max())
plt.gcf().autofmt_xdate()

plt.legend()
plt.grid(True)

# 保存图表
output_path = "/Users/athena/Desktop/本科毕设/4_建立水质预测模型_清水铺_ANN/Actual_vs_Predicted_Permanganate_Index_with_Warnings.png"
plt.savefig(output_path)
print(f"图表已保存到: {output_path}")

plt.show()