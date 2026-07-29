import pandas as pd
df = pd.read_csv("Data/Prediction/Control/metrics_results.csv")

# Keep only relevant columns (adjust names if needed)
df['ID_diff'] = abs(df['pred_id_count'] - df['gt_id_count'])
cols = ['HOTA.HOTA', 'Identity.IDF1', 'HOTA.AssA', 'ID_diff']
df = df[cols].dropna()

# === Compute full correlation matrix ===
corr_matrix = df.corr()

print("\n=== Full Correlation Matrix ===")
print(corr_matrix)

# === Correlation with ID_diff ===
corr_with_id = corr_matrix['ID_diff'].drop('ID_diff')

print("\n=== Correlation with ID_diff ===")
print(corr_with_id)

# === Rank metrics by absolute correlation ===
ranking = corr_with_id.abs().sort_values(ascending=False)

print("\n=== Metrics ranked by influence on ID_diff (correlation) ===")
print(ranking)

# === Optional: pretty interpretation ===
print("\n=== Interpretation ===")
for metric in ranking.index:
    val = corr_with_id[metric]
    direction = "decreases" if val < 0 else "increases"
    print(f"{metric}: stronger -> ID_diff {direction} (corr={val:.3f})")