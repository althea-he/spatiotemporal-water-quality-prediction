import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

# ==== 配置路径 ====
data_root = ""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 使用设备: {device}")

'''target_configs = {
    "总氮": {"hidden_dim": 256, "dropout": 0, "threshold": 0.5, "ylabel": "Total Nitrogen (mg/L)", "save_name": "Actual_vs_Predicted_Total_Nitrogen_with_Warnings.png"},
    "总磷": {"hidden_dim": 256, "dropout": 0, "threshold": 0.1, "ylabel": "Total Phosphorus (mg/L)", "save_name": "Actual_vs_Predicted_Total_Phosphorus_with_Warnings.png"},
    "氨氮": {"hidden_dim": 64, "dropout": 0, "threshold": 0.5, "ylabel": "Ammonia Nitrogen (mg/L)", "save_name": "Actual_vs_Predicted_Ammonia_Nitrogen_with_Warnings.png"}
}'''
target_configs = {
    "总磷": {"hidden_dim": 256, "dropout": 0, "threshold": 0.1, "ylabel": "Total Phosphorus (mg/L)", "save_name": "Actual_vs_Predicted_Total_Phosphorus_with_Warnings.png"},
}

WINDOW_SIZE = 7
BATCH_SIZE = 32
LAGS = [1, 2, 3, 7, 14, 30]

# ==== 滑动窗口函数 ====
def create_sliding_windows(graph_features, non_graph_tensor, target_tensor, window_size=3):
    Xg, Xn, Y = [], [], []
    for t in range(window_size, len(target_tensor)):
        Xg.append(graph_features[t - window_size:t])
        Xn.append(non_graph_tensor[t])
        Y.append(target_tensor[t])
    return torch.stack(Xg), torch.stack(Xn), torch.stack(Y)

# ==== Temporal Attention 模块 ====
class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_seq):
        Q = self.query_layer(x_seq)
        K = self.key_layer(x_seq)
        V = self.value_layer(x_seq)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(Q.shape[-1])
        attention_weights = self.softmax(attention_scores)
        attended = torch.bmm(attention_weights, V)
        out = attended.mean(dim=1)
        return out

# ==== STGCN+TemporalAttention模型 ====
class STGCN_Model(nn.Module):
    def __init__(self, in_channels, non_graph_dim, hidden_dim, dropout, window_size=3):
        super(STGCN_Model, self).__init__()
        self.window_size = window_size
        self.gats = nn.ModuleList([
            GATConv(in_channels, hidden_dim, heads=1, concat=False) for _ in range(window_size)
        ])
        self.temporal_att = TemporalAttention(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + non_graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_graph_seq, x_non_graph, edge_index):
        batch_size, window_size, num_nodes, in_channels = x_graph_seq.shape
        full_edge_index = torch.cat([edge_index + i * num_nodes for i in range(batch_size)], dim=1)
        x_graph_seq = x_graph_seq.view(batch_size * num_nodes, window_size, in_channels)

        x_time_list = []
        for t in range(self.window_size):
            x_t = x_graph_seq[:, t, :]
            x_gat = self.gats[t](x_t, full_edge_index)
            x_time_list.append(x_gat)

        x_cat = torch.stack(x_time_list, dim=1)
        x_cat = x_cat.view(batch_size, num_nodes, window_size, -1)[:, 0, :, :]
        x_att = self.temporal_att(x_cat)

        x_all = torch.cat([x_att, x_non_graph], dim=-1)
        x_out = self.mlp(x_all).squeeze(-1)
        return x_out

# ==== 主流程 ====
for target_variable, config in target_configs.items():
    print(f"\n▶ 当前预测目标: {target_variable}")

    selected_path = os.path.join(data_root, "石坎断面模型输入数据_selected_加污水.xlsx")
    edge_path = os.path.join(data_root, "Edge_List_Shikan_Subbasin10_ExactDistance.xlsx")

    df_selected = pd.read_excel(selected_path, sheet_name=target_variable)

    # ==== 特征工程 ====
    df_selected["日期"] = pd.to_datetime(df_selected["日期"])
    df_selected["weekday"] = df_selected["日期"].dt.weekday
    df_selected["month"] = df_selected["日期"].dt.month
    df_selected["quarter"] = df_selected["日期"].dt.quarter
    df_selected["dayofyear"] = df_selected["日期"].dt.dayofyear
    df_selected["dayofyear_sin"] = np.sin(2 * np.pi * df_selected["dayofyear"] / 365)
    df_selected["dayofyear_cos"] = np.cos(2 * np.pi * df_selected["dayofyear"] / 365)

    # 滞后特征
    for lag in LAGS:
        df_selected[f"{target_variable}_lag{lag}"] = df_selected[f"{target_variable}(mg/L)"].shift(lag)

    # 移动平均
    df_selected[f"{target_variable}_MA3"] = df_selected[f"{target_variable}(mg/L)"].rolling(window=3).mean()
    df_selected[f"{target_variable}_MA7"] = df_selected[f"{target_variable}(mg/L)"].rolling(window=7).mean()
    df_selected[f"{target_variable}_MA14"] = df_selected[f"{target_variable}(mg/L)"].rolling(window=14).mean()

    df_selected = df_selected.drop(columns=["dayofyear"]).dropna()

    # ==== 图特征 ====
    graph_bands = [col for col in df_selected.columns if col.startswith("b0")]
    non_graph_features = [col for col in df_selected.columns if col not in graph_bands + ["日期", f"{target_variable}(mg/L)"]]

    scaler_x = StandardScaler()
    non_graph_tensor = torch.tensor(scaler_x.fit_transform(df_selected[non_graph_features]), dtype=torch.float32)

    graph_features_list = []
    for band in graph_bands:
        band_path = os.path.join(data_root, f"MOD09GA_{band}_Shikan_Subbasin10.xlsx")
        df_band = pd.read_excel(band_path)
        band_tensor = torch.tensor(df_band.iloc[:, 1:].T.values, dtype=torch.float32)
        graph_features_list.append(band_tensor.unsqueeze(-1))
    graph_features = torch.cat(graph_features_list, dim=-1)

    graph_mean = graph_features.mean(dim=(0, 1), keepdim=True)
    graph_std = graph_features.std(dim=(0, 1), keepdim=True)
    graph_features = (graph_features - graph_mean) / (graph_std + 1e-6)

    edge_df = pd.read_excel(edge_path)
    grid_ids = sorted(edge_df["Grid_ID"].unique())
    num_nodes = len(grid_ids)
    K = 5

    from_idx, to_idx = [], []
    for i in range(num_nodes):
        if i != 0:
            from_idx.append(i)
            to_idx.append(0)
        for j in range(1, K+1):
            if i + j < num_nodes:
                from_idx.append(i)
                to_idx.append(i + j)
            if i - j >= 0:
                from_idx.append(i)
                to_idx.append(i - j)
        from_idx.append(i)
        to_idx.append(i)
    edge_index = torch.tensor([from_idx, to_idx], dtype=torch.long).to(device)

    scaler_y = StandardScaler()
    target_series = df_selected[f"{target_variable}(mg/L)"].values
    target_tensor = torch.tensor(scaler_y.fit_transform(target_series.reshape(-1, 1)), dtype=torch.float32)

    time_len = min(graph_features.shape[0], non_graph_tensor.shape[0], target_tensor.shape[0])
    graph_features = graph_features[:time_len]
    non_graph_tensor = non_graph_tensor[:time_len]
    target_tensor = target_tensor[:time_len]

    graph_seq, non_graph_seq, target_seq = create_sliding_windows(graph_features, non_graph_tensor, target_tensor, window_size=WINDOW_SIZE)

    full_dates = df_selected["日期"].iloc[WINDOW_SIZE:].reset_index(drop=True)
    split_date = pd.to_datetime('2022-06-01')

    # 智能分割
    all_train_mask = pd.to_datetime(full_dates) < split_date
    test_mask = pd.to_datetime(full_dates) >= split_date

    train_dates = full_dates[all_train_mask]
    Xg_train_cand = graph_seq[all_train_mask]
    Xn_train_cand = non_graph_seq[all_train_mask]
    y_train_cand = target_seq[all_train_mask]

    early_2022_mask = (pd.to_datetime(train_dates) >= pd.to_datetime('2022-01-01')) & (pd.to_datetime(train_dates) < split_date)
    np.random.seed(42)
    all_indices = np.arange(len(train_dates))
    np.random.shuffle(all_indices)
    early_2022_indices = np.where(early_2022_mask)[0]
    other_indices = np.setdiff1d(all_indices, early_2022_indices)

    final_train_indices = list(early_2022_indices) + list(other_indices)

    Xg_train_full = Xg_train_cand[final_train_indices]
    Xn_train_full = Xn_train_cand[final_train_indices]
    y_train_full = y_train_cand[final_train_indices]

    Xg_test = graph_seq[test_mask]
    Xn_test = non_graph_seq[test_mask]
    y_test = target_seq[test_mask]
    test_dates = full_dates[test_mask].reset_index(drop=True)

    train_dataset = TensorDataset(Xg_train_full, Xn_train_full, y_train_full)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = STGCN_Model(
        in_channels=graph_features.shape[-1],
        non_graph_dim=non_graph_tensor.shape[-1],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        window_size=WINDOW_SIZE
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    EPOCHS = 300
    for epoch in trange(EPOCHS, desc=f"Training {target_variable}"):
        model.train()
        running_loss = 0.0
        for batch_graph, batch_non_graph, batch_target in train_loader:
            batch_graph = batch_graph.to(device)
            batch_non_graph = batch_non_graph.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            outputs = model(batch_graph, batch_non_graph, edge_index)
            loss = loss_fn(outputs, batch_target.squeeze(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step(running_loss)

    model.eval()
    with torch.no_grad():
        y_pred_test = model(Xg_test.to(device), Xn_test.to(device), edge_index).cpu().numpy()

    y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

    tolerance = 0.05  # mg/L, small buffer to prevent false positives
    warning_mask = y_pred_test_inv > (config["threshold"] + tolerance)
    warning_dates = test_dates[warning_mask]
    warning_values = y_pred_test_inv[warning_mask]

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, label="Actual", color='blue', alpha=0.6)
    plt.plot(test_dates, y_pred_test_inv, label="Predicted", color='red', linestyle="dashed", alpha=0.8)

    if len(warning_dates) > 0:
        plt.scatter(warning_dates, warning_values, color='red', label="⚠️ Exceeded", marker='o', s=50, edgecolors='black')

    plt.axhline(y=config["threshold"], color='black', linestyle='--', linewidth=1, label=f"Threshold ({config['threshold']} mg/L)")
    plt.xlabel("Date")
    plt.ylabel(config["ylabel"])
    plt.title(f"Actual vs Predicted {config['ylabel']} with Warnings")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    output_path = os.path.join(data_root, config["save_name"])
    plt.savefig(output_path, dpi=300)
    print(f"✔ 图表已保存到: {output_path}")
    plt.close()
