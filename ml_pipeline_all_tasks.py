#!/usr/bin/env python3
"""
ml_pipeline_all_tasks.py

- Loads all summary CSVs recursively (TCP + UDP)
- Adds protocol_type column
- Trains/evaluates:
    1) congestion detection (binary)
    2) scenario classification (A/B/C)
    3) RTT regression
    4) Throughput regression
    5) Buffer recommendation regression
- Saves models, predictions CSV, feature-importance PNGs and a model_summary.json

Usage:
  python ml_pipeline_all_tasks_FIXED.py --data-dir ./ --out-model-dir ./models_combined_fixed --save-models --seed 42
"""
import os
import glob
import argparse
import joblib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -------------------------
# Helpers
# -------------------------
def load_all_summaries(data_dir):
    p = Path(data_dir)
    files = sorted(list(p.glob("**/summary*.csv")) + list(p.glob("**/*summary*.csv")))
    if not files:
        f = p / "summary.csv"
        if f.exists(): files = [f]
    if not files:
        raise FileNotFoundError(f"No summary CSVs found in {data_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    return df

def prepare_dataframe(df):
    # cleanup
    df.columns = [c.strip() for c in df.columns]
    # ensure protocol column
    df["protocol"] = df.get("protocol", "").astype(str).str.lower().fillna("")
    # if protocol not present but file name includes tcp/udp, try to infer
    df.loc[df["protocol"] == "", "protocol"] = df.loc[df["protocol"] == "", "__source_file"].str.lower().apply(lambda s: "tcp" if "tcp" in s else ("udp" if "udp" in s else ""))
    df["protocol_type"] = df["protocol"].apply(lambda x: "tcp" if "tcp" in str(x).lower() else ("udp" if "udp" in str(x).lower() else "unknown"))
    # scenario
    df["scenario"] = df.get("scenario", "").astype(str).str.upper().replace("", "A")
    df.loc[~df["scenario"].isin(["A","B","C"]), "scenario"] = "A"
    # numeric columns
    num_cols = ["avg_bps","total_bytes","total_packets","total_lost_or_retrans","packet_loss_rate","avg_jitter_ms","avg_rtt_ms","bdp_bytes","recommended_buffer_bytes","client_buffer_bytes","server_buffer_bytes"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # congestion_level: use existing else derive
    if "congestion_level" not in df.columns or df["congestion_level"].isnull().all():
        df["congestion_level"] = np.where((df["packet_loss_rate"] > 0.03) | (df["total_lost_or_retrans"] / df["total_packets"].replace(0,1) > 0.03), "high", "low")
    df["congestion_level"] = df["congestion_level"].astype(str)
    # features
    df["loss_ratio"] = df["total_lost_or_retrans"] / df["total_packets"].replace(0,1)
    df["avg_mbps"] = df["avg_bps"] / 1e6
    df["bdp_kb"] = df["bdp_bytes"] / 1024.0
    df["protocol_cat"] = df["protocol_type"].map({"tcp":1,"udp":0}).fillna(0).astype(int)
    # filter short duration
    if "duration" in df.columns:
        df = df[df["duration"].fillna(0).astype(float) >= 1]
    return df

def print_regression_metrics(y_true, y_pred, name="target"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    try:
        # newer sklearn versions
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # older sklearn versions
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} : R2={r2:.4f}, MAE={mae:.3f}, RMSE={rmse:.3f}")


def plot_feature_importances(importances, features, outpath, title):
    fi = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:30]
    names = [f for f,_ in fi]
    vals = [v for _,v in fi]
    plt.figure(figsize=(8, max(3, 0.3*len(names))))
    sns.barplot(x=vals, y=names)
    plt.title(title)
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------------
# Main (FIXED)
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./", help="Root folder containing summary CSVs")
    ap.add_argument("--out-model-dir", default="./models_combined_fixed", help="Output dir for models, plots, preds")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--predict-on-all", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_model_dir, exist_ok=True)

    print("Loading summary CSVs from:", args.data_dir)
    df = load_all_summaries(args.data_dir)
    print("Raw rows:", len(df))
    df = prepare_dataframe(df)
    print("Rows after prepare:", len(df))

    # train/test split
    strat = df["scenario"] if "scenario" in df.columns else None
    df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=strat)

    # --- Feature Lists (The FIX) ---
    # We define a separate, leak-free feature list for EACH task.
    all_cols = set(df.columns)

    # Task 1: Predict congestion. 
    # LEAKAGE: Do NOT use packet_loss_rate or total_lost_or_retrans, as they DEFINE congestion.
    # Goal: Predict congestion from *symptoms* like RTT, jitter, and throughput.
    features_1 = [f for f in [
        "protocol_cat", "avg_bps", "avg_jitter_ms", "avg_rtt_ms",
        "client_buffer_bytes", "server_buffer_bytes"
    ] if f in all_cols]

    # Task 2: Classify scenario.
    # This is the ONE task where using all metrics is valid.
    # Goal: Use all observed metrics to classify the *cause* (which scenario).
    features_2 = [f for f in [
        "protocol_cat", "avg_bps", "total_packets", "total_lost_or_retrans", 
        "packet_loss_rate", "avg_jitter_ms", "avg_rtt_ms", "bdp_bytes"
    ] if f in all_cols]

    # Task 3: Predict RTT.
    # LEAKAGE: Do NOT use avg_rtt_ms or bdp_bytes (which is calculated from RTT).
    # Goal: Predict RTT from protocol, throughput, loss, and jitter.
    features_3 = [f for f in [
        "protocol_cat", "avg_bps", "total_packets", "total_lost_or_retrans", 
        "packet_loss_rate", "avg_jitter_ms", 
        "client_buffer_bytes", "server_buffer_bytes"
    ] if f in all_cols]

    # Task 4: Predict Throughput.
    # LEAKAGE: Do NOT use avg_bps or bdp_bytes.
    # Goal: Predict throughput from protocol, RTT, loss, and jitter.
    features_4 = [f for f in [
        "protocol_cat", "total_packets", "total_lost_or_retrans", 
        "packet_loss_rate", "avg_jitter_ms", "avg_rtt_ms",
        "client_buffer_bytes", "server_buffer_bytes"
    ] if f in all_cols]

    # Task 5: Predict Buffer.
    # LEAKAGE: Do NOT use recommended_buffer_bytes or bdp_bytes.
    # Goal: Predict the *optimal* buffer using the raw metrics (RTT, throughput) that *create* the BDP.
    features_5 = [f for f in [
        "protocol_cat", "avg_bps", "avg_rtt_ms", "avg_jitter_ms", 
        "packet_loss_rate", "client_buffer_bytes", "server_buffer_bytes"
    ] if f in all_cols]
    
    # We also need a combined list for the final prediction CSV
    all_features_used = sorted(list(set(features_1 + features_2 + features_3 + features_4 + features_5)))
    
    print(f"\nTask 1 (Congestion) Features:\n{features_1}")
    print(f"\nTask 2 (Scenario) Features:\n{features_2}")
    print(f"\nTask 3 (RTT) Features:\n{features_3}")
    print(f"\nTask 4 (Throughput) Features:\n{features_4}")
    print(f"\nTask 5 (Buffer) Features:\n{features_5}")


    # Task 1: congestion detection
    print("\nTask 1: congestion detection (binary)")
    X1_train = df_train[features_1]; y1_train = df_train["congestion_level"]
    X1_test = df_test[features_1]; y1_test = df_test["congestion_level"]
    clf1 = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1))])
    clf1.fit(X1_train, y1_train)
    y1_pred = clf1.predict(X1_test)
    print(classification_report(y1_test, y1_pred))
    print("Confusion matrix:\n", confusion_matrix(y1_test, y1_pred))
    fi1 = clf1.named_steps["rf"].feature_importances_
    plot_feature_importances(fi1, features_1, os.path.join(args.out_model_dir, "fi_congestion.png"), "Feature importance - Congestion")

    # Task 2: scenario classification
    print("\nTask 2: scenario classification (A/B/C)")
    X2_train = df_train[features_2]; y2_train = df_train["scenario"]
    X2_test = df_test[features_2]; y2_test = df_test["scenario"]
    clf2 = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1))])
    clf2.fit(X2_train, y2_train)
    y2_pred = clf2.predict(X2_test)
    print(classification_report(y2_test, y2_pred))
    print("Confusion matrix:\n", confusion_matrix(y2_test, y2_pred))
    fi2 = clf2.named_steps["rf"].feature_importances_
    plot_feature_importances(fi2, features_2, os.path.join(args.out_model_dir, "fi_scenario.png"), "Feature importance - Scenario")

    # Task 3: RTT regression
    print("\nTask 3: RTT regression (avg_rtt_ms)")
    df_rtt = df[df["avg_rtt_ms"] > 0.0].copy()
    if len(df_rtt) >= 30:
        tr, te = train_test_split(df_rtt, test_size=args.test_size, random_state=args.seed, stratify=df_rtt["scenario"])
        Xr_train = tr[features_3]; yr_train = tr["avg_rtt_ms"]
        Xr_test = te[features_3]; yr_test = te["avg_rtt_ms"]
        reg_rtt = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=200, random_state=args.seed, n_jobs=-1))])
        reg_rtt.fit(Xr_train, yr_train)
        yr_pred = reg_rtt.predict(Xr_test)
        print_regression_metrics(yr_test, yr_pred, "avg_rtt_ms")
        fi_rtt = reg_rtt.named_steps["rf"].feature_importances_
        plot_feature_importances(fi_rtt, features_3, os.path.join(args.out_model_dir, "fi_rtt.png"), "Feature importance - RTT")
    else:
        print("Skipping RTT regression (not enough samples)."); reg_rtt=None

    # Task 4: throughput regression
    print("\nTask 4: Throughput regression (avg_bps)")
    trt, tet = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df["scenario"])
    Xt_train = trt[features_4]; yt_train = trt["avg_bps"]
    Xt_test = tet[features_4]; yt_test = tet["avg_bps"]
    reg_thr = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=200, random_state=args.seed, n_jobs=-1))])
    reg_thr.fit(Xt_train, yt_train)
    yt_pred = reg_thr.predict(Xt_test)
    print_regression_metrics(yt_test, yt_pred, "avg_bps")
    fi_thr = reg_thr.named_steps["rf"].feature_importances_
    plot_feature_importances(fi_thr, features_4, os.path.join(args.out_model_dir, "fi_throughput.png"), "Feature importance - Throughput")

    # Task 5: recommended buffer regression
    print("\nTask 5: Buffer recommendation regression")
    if "recommended_buffer_bytes" in df.columns and (df["recommended_buffer_bytes"] > 0).sum() >= 30:
        df_buf = df[df["recommended_buffer_bytes"] > 0].copy()
        tb, teb = train_test_split(df_buf, test_size=args.test_size, random_state=args.seed, stratify=df_buf["scenario"])
        Xb_train = tb[features_5]; yb_train = tb["recommended_buffer_bytes"]
        Xb_test = teb[features_5]; yb_test = teb["recommended_buffer_bytes"]
        reg_buf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=200, random_state=args.seed, n_jobs=-1))])
        reg_buf.fit(Xb_train, yb_train)
        yb_pred = reg_buf.predict(Xb_test)
        print_regression_metrics(yb_test, yb_pred, "recommended_buffer_bytes")
        fi_buf = reg_buf.named_steps["rf"].feature_importances_
        plot_feature_importances(fi_buf, features_5, os.path.join(args.out_model_dir, "fi_buffer.png"), "Feature importance - Buffer")
    else:
        print("Skipping buffer regression (no field or too few samples)."); reg_buf=None

    # Save models & predictions
    if args.save_models:
        joblib.dump(clf1, os.path.join(args.out_model_dir, "clf_congestion.joblib"))
        joblib.dump(clf2, os.path.join(args.out_model_dir, "clf_scenario.joblib"))
        joblib.dump(reg_thr, os.path.join(args.out_model_dir, "reg_throughput.joblib"))
        if 'reg_rtt' in locals() and reg_rtt is not None:
            joblib.dump(reg_rtt, os.path.join(args.out_model_dir, "reg_rtt.joblib"))
        if 'reg_buf' in locals() and reg_buf is not None:
            joblib.dump(reg_buf, os.path.join(args.out_model_dir, "reg_buffer.joblib"))
        print("Saved models in", args.out_model_dir)

    # Predictions on test set
    preds = df_test.copy()
    preds["pred_congestion"] = clf1.predict(df_test[features_1])
    preds["pred_scenario"] = clf2.predict(df_test[features_2])
    preds["pred_avg_bps"] = reg_thr.predict(df_test[features_4])
    if 'reg_rtt' in locals() and reg_rtt is not None:
        preds["pred_avg_rtt_ms"] = reg_rtt.predict(df_test[features_3])
    if 'reg_buf' in locals() and reg_buf is not None:
        preds["pred_recommended_buffer_bytes"] = reg_buf.predict(df_test[features_5])
    preds.to_csv(os.path.join(args.out_model_dir, "test_set_predictions.csv"), index=False)

    # model summary
    summary = {
        "n_rows": int(len(df)),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "features_tasks": {
            "congestion": features_1,
            "scenario": features_2,
            "rtt": features_3,
            "throughput": features_4,
            "buffer": features_5
        }
    }
    with open(os.path.join(args.out_model_dir, "model_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Done. Outputs in", args.out_model_dir)

if __name__ == "__main__":
    main()