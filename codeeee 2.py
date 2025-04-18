import os, re, glob, random, pandas as pd, numpy as np, matplotlib.pyplot as plt, shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def extract_data_for_files(sim_files, generator_ratings, row_offset=0, rocof_index=600):
    data_records = []
    for sim_csv in sim_files:
        basename = os.path.basename(sim_csv)
        match = re.search(r'cnt(\d+)', basename)
        if not match:
            continue
        file_number = int(match.group(1))
        row_index = file_number + row_offset
        try:
            df = pd.read_csv(sim_csv, header=None)
        except Exception:
            continue
        if df.shape[0] < 3:
            continue
        property_labels = df.iloc[1].astype(str).tolist()
        active_power_dict, generator_count = {}, 0
        for col_index, prop_label in enumerate(property_labels):
            if "active power" in prop_label.lower():
                generator_count += 1
                active_power_dict[f"G{generator_count}_ActivePower"] = df.iloc[2, col_index]
                if generator_count >= 10:
                    break
        freq_cols = [c for c in df.columns if "electrical frequency" in property_labels[c].lower()]
        freq_nadir = pd.to_numeric(df.iloc[2:, freq_cols].stack(), errors="coerce").min() if freq_cols else None
        rocof_cols = [c for c in df.columns if "derivative of el. frequency" in property_labels[c].lower()]
        rocof_value = df.iloc[rocof_index, rocof_cols].min() if rocof_cols and df.shape[0] > rocof_index else None
        if not (0 <= row_index < len(generator_ratings)):
            continue
        rating_row = generator_ratings.iloc[row_index]
        cig_value, loading_value = rating_row.get("cig", None), rating_row.get("loading", None)
        rating_dict = {f"G{i}_Rating": rating_row[f"g{i}"] for i in range(1, 11) if f"g{i}" in rating_row.index}
        record = {"Filename": basename, "cnt": file_number, "freq_nadir": freq_nadir, "rocof": rocof_value,
                  "CIG": cig_value, "LOADING": loading_value}
        record.update(active_power_dict)
        record.update(rating_dict)
        data_records.append(record)
    return data_records

def build_datasets():
    sim_files_folder = "/Users/raghavdewra/Desktop/dataset"
    rating_file_path = "/Users/raghavdewra/Desktop/GENERATOR_RATINGS.csv"
    train_csv = "/Users/raghavdewra/Desktop/refine/training_data.csv"
    test_csv = "/Users/raghavdewra/Desktop/refine/testing_data.csv"
    train_ratio, row_offset, rocof_index = 0.80, 0, 600
    if not os.path.exists(rating_file_path):
        raise FileNotFoundError(rating_file_path)
    generator_ratings = pd.read_csv(rating_file_path)
    generator_ratings.columns = generator_ratings.columns.str.lower().str.strip()
    sim_files = glob.glob(os.path.join(sim_files_folder, "*.csv"))
    if not sim_files:
        raise FileNotFoundError(sim_files_folder)
    random.shuffle(sim_files)
    cutoff = int(len(sim_files) * train_ratio)
    train_records = extract_data_for_files(sim_files[:cutoff], generator_ratings, row_offset, rocof_index)
    test_records = extract_data_for_files(sim_files[cutoff:], generator_ratings, row_offset, rocof_index)
    pd.DataFrame(train_records).to_csv(train_csv, index=False)
    pd.DataFrame(test_records).to_csv(test_csv, index=False)

def run_ml_pipeline():
    train_csv = "/Users/raghavdewra/Desktop/refine/training_data.csv"
    test_csv = "/Users/raghavdewra/Desktop/refine/testing_data.csv"
    df_train, df_test = pd.read_csv(train_csv), pd.read_csv(test_csv)
    ignore_cols = ["Filename", "cnt", "freq_nadir", "rocof"]
    feature_cols = [c for c in df_train.columns if c not in ignore_cols]
    X_train, X_test = df_train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0), \
                      df_test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train_freq, y_test_freq = pd.to_numeric(df_train["freq_nadir"], errors="coerce").fillna(df_train["freq_nadir"].mean()), \
                                pd.to_numeric(df_test["freq_nadir"], errors="coerce").fillna(df_test["freq_nadir"].mean())
    y_train_rocof, y_test_rocof = pd.to_numeric(df_train["rocof"], errors="coerce").fillna(df_train["rocof"].mean()), \
                                  pd.to_numeric(df_test["rocof"], errors="coerce").fillna(df_test["rocof"].mean())
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 8, 10, None],
                  "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]}
    rf_freq = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring="r2", cv=5, n_jobs=-1)
    rf_freq.fit(X_train_scaled, y_train_freq)
    best_freq = rf_freq.best_estimator_
    freq_pred = best_freq.predict(X_test_scaled)
    print("[FREQ]", round(np.sqrt(mean_squared_error(y_test_freq, freq_pred)), 4),
          round(r2_score(y_test_freq, freq_pred), 4),
          round(mean_absolute_error(y_test_freq, freq_pred), 4))
    rf_rocof = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring="r2", cv=5, n_jobs=-1)
    rf_rocof.fit(X_train_scaled, y_train_rocof)
    best_rocof = rf_rocof.best_estimator_
    rocof_pred = best_rocof.predict(X_test_scaled)
    print("[ROCOF]", round(np.sqrt(mean_squared_error(y_test_rocof, rocof_pred)), 4),
          round(r2_score(y_test_rocof, rocof_pred), 4),
          round(mean_absolute_error(y_test_rocof, rocof_pred), 4))
    pfi_freq = permutation_importance(best_freq, X_test_scaled, y_test_freq, scoring="r2", n_repeats=10, random_state=42)
    pd.DataFrame({"feature": feature_cols, "importance": pfi_freq.importances_mean}).sort_values("importance", ascending=False)\
        .plot.bar(x="feature", y="importance", figsize=(10, 6), rot=45); plt.tight_layout(); plt.savefig("pfi_freq_nadir.png"); plt.show()
    pfi_rocof = permutation_importance(best_rocof, X_test_scaled, y_test_rocof, scoring="r2", n_repeats=10, random_state=42)
    pd.DataFrame({"feature": feature_cols, "importance": pfi_rocof.importances_mean}).sort_values("importance", ascending=False)\
        .plot.bar(x="feature", y="importance", figsize=(10, 6), rot=45); plt.tight_layout(); plt.savefig("pfi_rocof.png"); plt.show()
    plt.figure(figsize=(8, 6)); plt.scatter(y_test_freq, freq_pred, alpha=0.7, edgecolor="k")
    plt.plot([y_test_freq.min(), y_test_freq.max()], [y_test_freq.min(), y_test_freq.max()], "r--", lw=2)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Frequency Nadir"); plt.tight_layout()
    plt.savefig("actual_vs_predicted_freq_nadir.png"); plt.show()
    plt.figure(figsize=(8, 6)); plt.scatter(y_test_rocof, rocof_pred, alpha=0.7, edgecolor="k")
    plt.plot([y_test_rocof.min(), y_test_rocof.max()], [y_test_rocof.min(), y_test_rocof.max()], "r--", lw=2)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("RoCoF"); plt.tight_layout()
    plt.savefig("actual_vs_predicted_rocof.png"); plt.show()
    plt.figure(figsize=(8, 6)); plt.hist(freq_pred - y_test_freq, bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel("Error"); plt.ylabel("Frequency"); plt.title("Frequency Nadir Errors"); plt.tight_layout()
    plt.savefig("error_distribution_freq_nadir.png"); plt.show()
    plt.figure(figsize=(8, 6)); plt.hist(rocof_pred - y_test_rocof, bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel("Error"); plt.ylabel("Frequency"); plt.title("RoCoF Errors"); plt.tight_layout()
    plt.savefig("error_distribution_rocof.png"); plt.show()
    explainer = shap.Explainer(best_freq, X_train_scaled); shap_values = explainer(X_test_scaled)
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_cols, max_display=15, show=False)
    plt.title("SHAP Frequency Nadir"); plt.tight_layout(); plt.savefig("shap_summary_freq_nadir.png"); plt.show()
    explainer_r = shap.Explainer(best_rocof, X_train_scaled); shap_values_r = explainer_r(X_test_scaled)
    shap.summary_plot(shap_values_r, features=X_test, feature_names=feature_cols, max_display=15, show=False)
    plt.title("SHAP RoCoF"); plt.tight_layout(); plt.savefig("shap_summary_rocof.png"); plt.show()

def main():
    build_datasets()
    run_ml_pipeline()

if __name__ == "__main__":
    main()
