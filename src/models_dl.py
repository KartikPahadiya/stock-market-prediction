"""
==============================================================
AI Stock Prediction System
Pooled Deep Learning Models
Trains ONE model across all tickers with ticker embeddings.
All parameters now come from config.yaml via config_loader.
==============================================================
"""
import warnings
warnings.filterwarnings("ignore")

import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from config_loader import data_path, RANDOM_STATE, split_ratios, dl_model_params

# ==========================================================
# Paths (from config)
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = data_path("processed_features")
MODEL_DIR = data_path("models") / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = data_path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR = data_path("processed_stocks").parent / "predictions_dl"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Config-derived parameters
# ==========================================================

TARGET = "Target_Return"
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = split_ratios()
DL = dl_model_params()

SEQUENCE_LENGTH = DL.get("sequence_length", 60)
BATCH_SIZE = DL.get("batch_size", 32)
EPOCHS = DL.get("epochs", 50)
LEARNING_RATE = DL.get("learning_rate", 0.001)
TICKER_EMBED_DIM = DL.get("embedding_dim", 8)
PATIENCE = DL.get("patience", 7)

HIDDEN_SIZES = DL.get("hidden_units", {})
HIDDEN_SIZE = HIDDEN_SIZES.get("LSTM", 64)
NUM_LAYERS = 2
DROPOUT = 0.2

RANDOM_STATE = RANDOM_STATE

# ==========================================================
# Reproducibility helpers
# ==========================================================

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing Device : {DEVICE}")


# ==========================================================
# Metrics
# ==========================================================

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": rmse, "MAE": mae, "MAPE": mape,
        "R2": r2, "Directional_Accuracy": directional_accuracy(y_true, y_pred)
    }


# ==========================================================
# Data Loading
# ==========================================================

def load_stock_dataset(file):
    print("=" * 70)
    print(f"Loading {file.name}")
    print("=" * 70)
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.reset_index(drop=True, inplace=True)
    return df


def train_test_split_time(df):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train = df.iloc[:train_end].copy().reset_index(drop=True)
    val = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test = df.iloc[val_end:].copy().reset_index(drop=True)
    return train, val, test


# ==========================================================
# Feature Preparation (Pooled)
# ==========================================================

def prepare_features_pooled(train, val, test):
    drop_columns = ["Date", "ticker", "Target_Close", "Target_Return", "Target_Direction"]
    X_train = train.drop(columns=drop_columns, errors="ignore")
    X_val = val.drop(columns=drop_columns, errors="ignore")
    X_test = test.drop(columns=drop_columns, errors="ignore")

    y_train = train[TARGET]
    y_val = val[TARGET]
    y_test = test[TARGET]

    train_mask = X_train.notnull().all(axis=1)
    val_mask = X_val.notnull().all(axis=1)
    test_mask = X_test.notnull().all(axis=1)

    for name, mask in [("train", ~train_mask), ("val", ~val_mask), ("test", ~test_mask)]:
        if mask.sum() > 0:
            print(f"WARNING: dropping {mask.sum()} NaN rows in {name}")

    X_train = X_train[train_mask]; y_train = y_train[train_mask]
    X_val = X_val[val_mask]; y_val = y_val[val_mask]
    X_test = X_test[test_mask]; y_test = y_test[test_mask]

    if "ticker_id" not in X_train.columns:
        raise KeyError("ticker_id column missing -- did you forget to add it before splitting?")

    ticker_train = X_train["ticker_id"].values.astype(np.int64)
    ticker_val = X_val["ticker_id"].values.astype(np.int64)
    ticker_test = X_test["ticker_id"].values.astype(np.int64)

    feature_cols = [c for c in X_train.columns if c != "ticker_id"]

    X_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X_train[feature_cols])
    X_val_scaled = X_scaler.transform(X_val[feature_cols])
    X_test_scaled = X_scaler.transform(X_test[feature_cols])

    X_train = np.column_stack([X_train_scaled, ticker_train])
    X_val = np.column_stack([X_val_scaled, ticker_val])
    X_test = np.column_stack([X_test_scaled, ticker_test])

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler


# ==========================================================
# Sequence Creation
# ==========================================================

def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders_pooled(train_df, val_df, test_df):
    X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler = prepare_features_pooled(
        train_df, val_df, test_df
    )

    def seq_per_ticker(X_arr, y_arr):
        X_seqs, y_seqs = [], []
        ticker_ids = X_arr[:, -1].astype(int)
        for tid in np.unique(ticker_ids):
            mask = ticker_ids == tid
            X_t = X_arr[mask]
            y_t = y_arr[mask]
            if len(X_t) <= SEQUENCE_LENGTH:
                continue
            X_s, y_s = create_sequences(X_t, y_t, SEQUENCE_LENGTH)
            X_seqs.append(X_s)
            y_seqs.append(y_s)
        if len(X_seqs) == 0:
            return np.array([]), np.array([])
        return np.concatenate(X_seqs), np.concatenate(y_seqs)

    X_train, y_train = seq_per_ticker(X_train, y_train)
    X_val, y_val = seq_per_ticker(X_val, y_val)
    X_test, y_test = seq_per_ticker(X_test, y_test)

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[2] - 1
    return train_loader, val_loader, test_loader, X_scaler, y_scaler, input_size


# ==========================================================
# Model Classes (with Ticker Embedding)
# ==========================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, num_tickers=10, ticker_embed_dim=TICKER_EMBED_DIM,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.ticker_embed = nn.Embedding(num_tickers, ticker_embed_dim)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size + ticker_embed_dim, 1)

    def forward(self, x):
        ticker_ids = x[:, -1, -1].long()
        features = x[:, :, :-1]
        embed = self.ticker_embed(ticker_ids)
        out, _ = self.lstm(features)
        out = out[:, -1, :]
        out = torch.cat([out, embed], dim=-1)
        return self.fc(out).squeeze()


class GRUModel(nn.Module):
    def __init__(self, input_size, num_tickers=10, ticker_embed_dim=TICKER_EMBED_DIM,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.ticker_embed = nn.Embedding(num_tickers, ticker_embed_dim)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size + ticker_embed_dim, 1)

    def forward(self, x):
        ticker_ids = x[:, -1, -1].long()
        features = x[:, :, :-1]
        embed = self.ticker_embed(ticker_ids)
        out, _ = self.gru(features)
        out = out[:, -1, :]
        out = torch.cat([out, embed], dim=-1)
        return self.fc(out).squeeze()


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, num_tickers=10, ticker_embed_dim=TICKER_EMBED_DIM,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.ticker_embed = nn.Embedding(num_tickers, ticker_embed_dim)
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 + ticker_embed_dim, 1)

    def forward(self, x):
        ticker_ids = x[:, -1, -1].long()
        features = x[:, :, :-1]
        embed = self.ticker_embed(ticker_ids)
        out, _ = self.bilstm(features)
        out = out[:, -1, :]
        out = torch.cat([out, embed], dim=-1)
        return self.fc(out).squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_tickers=10, ticker_embed_dim=TICKER_EMBED_DIM,
                 d_model=64, nhead=4, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.ticker_embed = nn.Embedding(num_tickers, ticker_embed_dim)
        self.embedding = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model + ticker_embed_dim, 1)

    def forward(self, x):
        ticker_ids = x[:, -1, -1].long()
        features = x[:, :, :-1]
        embed = self.ticker_embed(ticker_ids)
        x = self.embedding(features)
        x = self.position(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = torch.cat([x, embed], dim=-1)
        return self.fc(x).squeeze()


MODELS = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "BiLSTM": BiLSTMModel,
    "Transformer": TransformerModel
}


# ==========================================================
# Loss & Training
# ==========================================================

criterion = nn.SmoothL1Loss()


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader):
    model.eval()
    total_loss = 0
    predictions, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    return total_loss / len(loader), np.array(actuals), np.array(predictions)


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.stop = False
        self.best_state = None

    def __call__(self, loss, model=None):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            if model is not None:
                self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ==========================================================
# Pooled Evaluation
# ==========================================================

def evaluate_pooled_model(model, test_loader, y_scaler, id_to_ticker, model_name):
    model.eval()
    all_ticker_ids, all_actuals_scaled, all_preds_scaled = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            outputs = model(X)
            ticker_ids = X[:, -1, -1].cpu().long().numpy()
            all_ticker_ids.extend(ticker_ids)
            all_actuals_scaled.extend(y.cpu().numpy())
            all_preds_scaled.extend(outputs.cpu().numpy())

    all_actuals = y_scaler.inverse_transform(np.array(all_actuals_scaled).reshape(-1, 1)).flatten()
    all_preds = y_scaler.inverse_transform(np.array(all_preds_scaled).reshape(-1, 1)).flatten()

    results = []
    z = evaluate_model(all_actuals, np.zeros_like(all_actuals))
    z.update({"Ticker": "ALL", "Model": "ZeroBaseline"})
    results.append(z)

    m = evaluate_model(all_actuals, all_preds)
    m.update({"Ticker": "ALL", "Model": model_name})
    results.append(m)

    for tid, ticker_name in id_to_ticker.items():
        mask = np.array(all_ticker_ids) == tid
        if mask.sum() == 0:
            continue
        actual = all_actuals[mask]
        pred = all_preds[mask]
        mm = evaluate_model(actual, pred)
        mm.update({"Ticker": ticker_name, "Model": model_name})
        results.append(mm)
        zz = evaluate_model(actual, np.zeros_like(actual))
        zz.update({"Ticker": ticker_name, "Model": "ZeroBaseline"})
        results.append(zz)

    return results


# ==========================================================
# Train One Architecture
# ==========================================================

def train_model(model_name, model, train_loader, val_loader, test_loader, y_scaler, id_to_ticker):
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stop = EarlyStopping(patience=PATIENCE)

    best_loss = np.inf
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, _, _ = validate(model, val_loader)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1:03d} Train:{train_loss:.4f} Val:{val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}.pth")

        early_stop(val_loss, model)
        if early_stop.stop:
            print("Early Stopping")
            break

    early_stop.restore_best(model)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title(f"{model_name} - Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(REPORT_DIR / f"{model_name}_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    results = evaluate_pooled_model(model, test_loader, y_scaler, id_to_ticker, model_name)

    print(f"\n{model_name} -- per-ticker RMSE:")
    for r in results:
        if r["Ticker"] != "ALL" and r["Model"] == model_name:
            print(f"  {r['Ticker']}: RMSE={r['RMSE']:.6f}  DirAcc={r['Directional_Accuracy']:.1f}%")

    return results


# ==========================================================
# Main
# ==========================================================

def main():
    set_seeds(RANDOM_STATE)

    feature_files = sorted(FEATURE_DIR.glob("*.csv"))
    print(f"\nFound {len(feature_files)} datasets.\n")

    ticker_names = [f.stem.replace("_features", "") for f in feature_files]
    ticker_to_id = {name: i for i, name in enumerate(ticker_names)}
    id_to_ticker = {i: name for name, i in ticker_to_id.items()}
    print(f"Tickers: {ticker_names}\n")

    all_train, all_val, all_test = [], [], []
    for file in feature_files:
        df = load_stock_dataset(file)
        ticker_name = file.stem.replace("_features", "")
        df["ticker_id"] = ticker_to_id[ticker_name]
        train_df, val_df, test_df = train_test_split_time(df)
        all_train.append(train_df)
        all_val.append(val_df)
        all_test.append(test_df)
        print(f"  {ticker_name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)

    print(f"\nPooled: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_loader, val_loader, test_loader, X_scaler, y_scaler, input_size = create_dataloaders_pooled(
        train_df, val_df, test_df
    )

    print(f"\nInput features: {input_size}")
    print(f"Num tickers:    {len(ticker_names)}")
    print(f"Train sequences: {len(train_loader.dataset)}")
    print(f"Val sequences:   {len(val_loader.dataset)}")
    print(f"Test sequences:  {len(test_loader.dataset)}")

    joblib.dump(X_scaler, MODEL_DIR / "pooled_X_scaler.pkl")
    joblib.dump(y_scaler, MODEL_DIR / "pooled_y_scaler.pkl")

    all_results = []
    for model_name, ModelClass in MODELS.items():
        # Use architecture-specific hidden size if available
        hidden = HIDDEN_SIZES.get(model_name, HIDDEN_SIZE)
        if model_name == "Transformer":
            # hidden is the full Transformer config dict from config.yaml
            d_model = hidden.get("d_model", 64)
            nhead = hidden.get("nhead", 4)
            num_layers = hidden.get("num_layers", 2)
            model = ModelClass(input_size=input_size, num_tickers=len(ticker_names),
                               d_model=d_model, nhead=nhead, num_layers=num_layers)
        else:
            model = ModelClass(input_size=input_size, num_tickers=len(ticker_names), hidden_size=hidden)
        results = train_model(model_name, model, train_loader, val_loader, test_loader, y_scaler, id_to_ticker)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORT_DIR / "dl_results_pooled.csv", index=False)

    trained_only = results_df[~results_df["Model"].isin(["ZeroBaseline"])]
    best_models = trained_only.sort_values(["Ticker", "RMSE"]).groupby("Ticker", as_index=False).first()
    best_models.to_csv(REPORT_DIR / "best_dl_models_pooled.csv", index=False)

    print("\n" + "=" * 80)
    print("BEST DEEP LEARNING MODELS PER TICKER (Pooled Training)")
    print("=" * 80)
    print(best_models[["Ticker", "Model", "RMSE", "R2", "Directional_Accuracy"]])
    print("\nPooled Deep Learning Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
