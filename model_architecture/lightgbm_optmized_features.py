import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Path to the dataset
data_dir = r'C:\Users\admin\Desktop\DR2\11 All Datasets\01 NASA PCoE Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4'

print("="*80)
print("STEP 2: FEATURE SELECTION EFFECT ANALYSIS - LIGHTGBM (FIXED)")
print("="*80)

def extract_18_features(cycle_data, battery_id):
    """Extract the original 18 features from a discharge cycle"""
    features = {}
    
    voltage = cycle_data.Voltage_measured
    current = cycle_data.Current_measured
    temp = cycle_data.Temperature_measured
    time = cycle_data.Time
    
    # ============================================================
    # 1. VOLTAGE FEATURES
    # ============================================================
    features['voltage_mean'] = np.mean(voltage)
    features['voltage_std'] = np.std(voltage)
    features['voltage_min'] = np.min(voltage)
    features['voltage_max'] = np.max(voltage)
    features['voltage_range'] = np.max(voltage) - np.min(voltage)
    
    # Voltage slope
    if len(time) > 1:
        slope = np.polyfit(time, voltage, 1)[0]
        features['voltage_slope'] = abs(slope)
    
    # Voltage at specific times
    if len(time) >= 100:
        idx_10 = int(0.1 * len(time))
        idx_50 = int(0.5 * len(time))
        idx_90 = int(0.9 * len(time))
        features['voltage_at_10pct'] = voltage[idx_10]
        features['voltage_at_50pct'] = voltage[idx_50]
        features['voltage_at_90pct'] = voltage[idx_90]
    
    # ============================================================
    # 2. CURRENT FEATURES
    # ============================================================
    features['current_mean'] = np.mean(current)
    features['current_std'] = np.std(current)
    
    # ============================================================
    # 3. TEMPERATURE FEATURES
    # ============================================================
    features['temp_mean'] = np.mean(temp)
    features['temp_std'] = np.std(temp)
    features['temp_max'] = np.max(temp)
    features['temp_min'] = np.min(temp)
    features['temp_range'] = np.max(temp) - np.min(temp)
    features['temp_increase'] = temp[-1] - temp[0]
    
    # ============================================================
    # 4. TIME FEATURES
    # ============================================================
    features['discharge_time'] = time[-1]
    
    return features

def load_battery_data(file_path):
    """Load and process battery data with 18 features"""
    data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    battery_key = os.path.basename(file_path).replace('.mat', '')
    battery_data = data[battery_key]
    
    if hasattr(battery_data, 'cycle'):
        battery = battery_data.cycle
    else:
        battery = battery_data
    
    all_features = []
    
    for cycle in battery:
        if cycle.type == 'discharge':
            try:
                features = extract_18_features(cycle.data, battery_key)
                
                if hasattr(cycle.data, 'Capacity'):
                    capacity = float(cycle.data.Capacity)
                    features['capacity'] = capacity
                    features['soh'] = capacity / 2.0
                    all_features.append(features)
            except Exception as e:
                continue
    
    return pd.DataFrame(all_features)

print("\nLoading and processing battery data...")

# Process all batteries
all_data = []
for file_name in ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']:
    file_path = os.path.join(data_dir, file_name)
    print(f"  Processing: {file_name}")
    df = load_battery_data(file_path)
    df['battery_id'] = file_name.replace('.mat', '')
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal discharge cycles extracted: {len(df_all)}")

# Create SOC target
def compute_soc(group):
    min_cap = group.min()
    max_cap = group.max()
    return (group - min_cap) / (max_cap - min_cap) * 0.8 + 0.1

df_all['soc'] = df_all.groupby('battery_id')['capacity'].transform(compute_soc)

print(f"Total cycles with complete data: {len(df_all)}")

# Prepare features
feature_columns = [col for col in df_all.columns if col not in ['capacity', 'soh', 'soc', 'battery_id']]
X = df_all[feature_columns]
y_soh = df_all['soh']
y_soc = df_all['soc']
battery_ids = df_all['battery_id']

print(f"\nTotal features: {len(feature_columns)}")

# ============================================================
# STEP 1: GET FEATURE IMPORTANCE USING ALL DATA
# ============================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS (TRAINING ON ALL DATA)")
print("="*80)

# Scale all data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train LightGBM on all data for feature importance
model_soh = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_gain_to_split=0.01,
    random_state=42,
    verbose=-1
)
model_soh.fit(X_scaled, y_soh)

model_soc = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_gain_to_split=0.01,
    random_state=42,
    verbose=-1
)
model_soc.fit(X_scaled, y_soc)

# Get feature importance and NORMALIZE to sum to 1
importance_soh_raw = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_soh.feature_importances_
})
importance_soh_raw['importance_norm'] = importance_soh_raw['importance'] / importance_soh_raw['importance'].sum()
importance_soh = importance_soh_raw.sort_values('importance_norm', ascending=False)
importance_soh['cumulative'] = importance_soh['importance_norm'].cumsum()

importance_soc_raw = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_soc.feature_importances_
})
importance_soc_raw['importance_norm'] = importance_soc_raw['importance'] / importance_soc_raw['importance'].sum()
importance_soc = importance_soc_raw.sort_values('importance_norm', ascending=False)
importance_soc['cumulative'] = importance_soc['importance_norm'].cumsum()

print("\nSOH Feature Importance (Normalized):")
print("="*70)
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<15} {'Cumulative':<15}")
print("-"*70)
for i in range(len(importance_soh)):
    row = importance_soh.iloc[i]
    if row['importance_norm'] > 0.001:  # Only show features with >0.1% importance
        print(f"{i+1:<6} {row['feature']:<25} {row['importance_norm']:.4f}    {row['cumulative']:.4f}")

print("\nSOC Feature Importance (Normalized):")
print("="*70)
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<15} {'Cumulative':<15}")
print("-"*70)
for i in range(len(importance_soc)):
    row = importance_soc.iloc[i]
    if row['importance_norm'] > 0.001:
        print(f"{i+1:<6} {row['feature']:<25} {row['importance_norm']:.4f}    {row['cumulative']:.4f}")

# ============================================================
# STEP 2: FEATURE SELECTION ANALYSIS
# ============================================================
print("\n" + "="*80)
print("FEATURE SELECTION ANALYSIS")
print("="*80)

# Determine feature counts for different cumulative thresholds
thresholds = [0.90, 0.95, 0.98, 0.99]
feature_counts = {}

print("\nFeatures needed for cumulative importance:")
print("-"*60)
for threshold in thresholds:
    n_soh = (importance_soh['cumulative'] < threshold).sum() + 1
    n_soc = (importance_soc['cumulative'] < threshold).sum() + 1
    feature_counts[threshold] = {'soh': n_soh, 'soc': n_soc}
    print(f"  {threshold*100:.0f}%: SOH={n_soh} features, SOC={n_soc} features")

# ============================================================
# STEP 3: EVALUATE DIFFERENT FEATURE SETS
# ============================================================
print("\n" + "="*80)
print("EVALUATING DIFFERENT FEATURE SETS")
print("="*80)

# Define feature sets to test
feature_sets = {
    'All 18 Features': feature_columns,
    'Top 5 SOH': importance_soh.head(5)['feature'].tolist(),
    'Top 8 SOH': importance_soh.head(8)['feature'].tolist(),
    'Top 10 SOH': importance_soh.head(10)['feature'].tolist(),
    'Top 5 SOC': importance_soc.head(5)['feature'].tolist(),
    'Top 8 SOC': importance_soc.head(8)['feature'].tolist(),
    'Top 10 SOC': importance_soc.head(10)['feature'].tolist(),
    'Combined Top 5': list(set(
        importance_soh.head(5)['feature'].tolist() +
        importance_soc.head(5)['feature'].tolist()
    )),
    'Combined Top 8': list(set(
        importance_soh.head(8)['feature'].tolist() +
        importance_soc.head(8)['feature'].tolist()
    )),
    'Combined Top 10': list(set(
        importance_soh.head(10)['feature'].tolist() +
        importance_soc.head(10)['feature'].tolist()
    )),
}

# LightGBM fixed parameters
lgb_params = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_gain_to_split': 0.01,
    'random_state': 42,
    'verbose': -1
}

batteries = ['B0005', 'B0006', 'B0007', 'B0018']
results = []

for set_name, features in feature_sets.items():
    print(f"\nTesting: {set_name} ({len(features)} features)")
    
    # Filter to available features
    available_features = [f for f in features if f in df_all.columns]
    if not available_features or len(available_features) < 2:
        continue
    
    X_subset = df_all[available_features]
    
    soh_mape_list = []
    soc_mape_list = []
    soh_r2_list = []
    soc_r2_list = []
    
    # LOBO CV
    for test_battery in batteries:
        train_batteries = [b for b in batteries if b != test_battery]
        
        train_mask = df_all['battery_id'].isin(train_batteries)
        test_mask = df_all['battery_id'] == test_battery
        
        X_train = X_subset[train_mask]
        X_test = X_subset[test_mask]
        y_soh_train = y_soh[train_mask]
        y_soh_test = y_soh[test_mask]
        y_soc_train = y_soc[train_mask]
        y_soc_test = y_soc[test_mask]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # SOH Model
        model_soh_cv = lgb.LGBMRegressor(**lgb_params)
        model_soh_cv.fit(X_train_scaled, y_soh_train)
        
        # SOC Model
        model_soc_cv = lgb.LGBMRegressor(**lgb_params)
        model_soc_cv.fit(X_train_scaled, y_soc_train)
        
        pred_soh = model_soh_cv.predict(X_test_scaled)
        pred_soc = model_soc_cv.predict(X_test_scaled)
        
        mape_soh = np.mean(np.abs((y_soh_test - pred_soh) / y_soh_test)) * 100
        mape_soc = np.mean(np.abs((y_soc_test - pred_soc) / y_soc_test)) * 100
        r2_soh = r2_score(y_soh_test, pred_soh)
        r2_soc = r2_score(y_soc_test, pred_soc)
        
        soh_mape_list.append(mape_soh)
        soc_mape_list.append(mape_soc)
        soh_r2_list.append(r2_soh)
        soc_r2_list.append(r2_soc)
    
    results.append({
        'Feature Set': set_name,
        'Features': len(available_features),
        'SOH MAPE Avg': np.mean(soh_mape_list),
        'SOH MAPE Std': np.std(soh_mape_list),
        'SOC MAPE Avg': np.mean(soc_mape_list),
        'SOC MAPE Std': np.std(soc_mape_list),
        'SOH R² Avg': np.mean(soh_r2_list),
        'SOC R² Avg': np.mean(soc_r2_list)
    })

# ============================================================
# DISPLAY RESULTS
# ============================================================
print("\n" + "="*80)
print("FEATURE SELECTION COMPARISON RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('SOH MAPE Avg')

print("\nResults Table (sorted by SOH MAPE):")
print("="*120)
print(f"{'Feature Set':<20} {'Features':<10} {'SOH MAPE':<15} {'SOH R²':<12} {'SOC MAPE':<15} {'SOC R²':<12}")
print("-"*120)
for _, row in results_df.iterrows():
    print(f"{row['Feature Set']:<20} {row['Features']:<10} {row['SOH MAPE Avg']:<15.2f} {row['SOH R² Avg']:<12.4f} {row['SOC MAPE Avg']:<15.2f} {row['SOC R² Avg']:<12.4f}")

# ============================================================
# FIND BEST FEATURE SETS
# ============================================================
print("\n" + "="*80)
print("BEST FEATURE SETS")
print("="*80)

best_soh = results_df.loc[results_df['SOH MAPE Avg'].idxmin()]
best_soc = results_df.loc[results_df['SOC MAPE Avg'].idxmin()]

# Find best balanced (minimize both)
results_df['Score'] = (results_df['SOH MAPE Avg'] / results_df['SOH MAPE Avg'].min() + 
                       results_df['SOC MAPE Avg'] / results_df['SOC MAPE Avg'].min()) / 2
best_balanced = results_df.loc[results_df['Score'].idxmin()]

print(f"\nBest for SOH: {best_soh['Feature Set']}")
print(f"  Features: {best_soh['Features']}")
print(f"  SOH MAPE: {best_soh['SOH MAPE Avg']:.2f}%")
print(f"  SOC MAPE: {best_soh['SOC MAPE Avg']:.2f}%")

print(f"\nBest for SOC: {best_soc['Feature Set']}")
print(f"  Features: {best_soc['Features']}")
print(f"  SOH MAPE: {best_soc['SOH MAPE Avg']:.2f}%")
print(f"  SOC MAPE: {best_soc['SOC MAPE Avg']:.2f}%")

print(f"\nBest Balanced: {best_balanced['Feature Set']}")
print(f"  Features: {best_balanced['Features']}")
print(f"  SOH MAPE: {best_balanced['SOH MAPE Avg']:.2f}%")
print(f"  SOC MAPE: {best_balanced['SOC MAPE Avg']:.2f}%")

# ============================================================
# RECOMMENDED FEATURE SET
# ============================================================
print("\n" + "="*80)
print("RECOMMENDED FEATURE SET FOR DEPLOYMENT")
print("="*80)

# Get the features for the best balanced set
best_features = []
for fs in feature_sets:
    if fs == best_balanced['Feature Set']:
        best_features = feature_sets[fs]
        break

if not best_features:
    best_features = feature_sets['Combined Top 8']

print(f"\nRecommended: {best_balanced['Feature Set']}")
print(f"Number of features: {len(best_features)}")
print("\nFeatures:")
for i, f in enumerate(sorted(best_features)):
    print(f"  {i+1}. {f}")

print(f"\nExpected SOH MAPE: {best_balanced['SOH MAPE Avg']:.2f}%")
print(f"Expected SOC MAPE: {best_balanced['SOC MAPE Avg']:.2f}%")
print(f"Feature Reduction: 18 → {len(best_features)} features ({((18 - len(best_features)) / 18 * 100):.1f}% reduction)")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SOH Performance vs Features
ax1 = axes[0, 0]
ax1.plot(results_df['Features'], results_df['SOH MAPE Avg'], 'bo-', linewidth=2, markersize=8)
ax1.fill_between(results_df['Features'], 
                 results_df['SOH MAPE Avg'] - results_df['SOH MAPE Std'],
                 results_df['SOH MAPE Avg'] + results_df['SOH MAPE Std'], 
                 alpha=0.2)
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('SOH MAPE (%)')
ax1.set_title('SOH Performance vs Feature Count')
ax1.grid(True, alpha=0.3)

# SOC Performance vs Features
ax2 = axes[0, 1]
ax2.plot(results_df['Features'], results_df['SOC MAPE Avg'], 'ro-', linewidth=2, markersize=8)
ax2.fill_between(results_df['Features'], 
                 results_df['SOC MAPE Avg'] - results_df['SOC MAPE Std'],
                 results_df['SOC MAPE Avg'] + results_df['SOC MAPE Std'], 
                 alpha=0.2)
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('SOC MAPE (%)')
ax2.set_title('SOC Performance vs Feature Count')
ax2.grid(True, alpha=0.3)

# Combined Feature Importance
ax3 = axes[1, 0]
combined_importance = pd.merge(importance_soh, importance_soc, on='feature', suffixes=('_soh', '_soc'))
combined_importance['avg_importance'] = (combined_importance['importance_norm_soh'] + combined_importance['importance_norm_soc']) / 2
combined_importance = combined_importance.sort_values('avg_importance', ascending=False)

# Highlight recommended features
colors = ['gold' if f in best_features else 'lightblue' for f in combined_importance['feature'].head(15)]
ax3.barh(combined_importance['feature'].head(15), combined_importance['avg_importance'].head(15), color=colors)
ax3.set_xlabel('Average Importance (SOH + SOC)')
ax3.set_title('Top Features (Gold = Recommended)')
ax3.grid(True, alpha=0.3)

# Feature Reduction Benefit
ax4 = axes[1, 1]
baseline_soh = results_df[results_df['Feature Set'] == 'All 18 Features']['SOH MAPE Avg'].values[0]
baseline_soc = results_df[results_df['Feature Set'] == 'All 18 Features']['SOC MAPE Avg'].values[0]

reduction = [(18 - f) / 18 * 100 for f in results_df['Features']]
soh_improvement = [(baseline_soh - s) / baseline_soh * 100 for s in results_df['SOH MAPE Avg']]
soc_improvement = [(baseline_soc - s) / baseline_soc * 100 for s in results_df['SOC MAPE Avg']]

ax4.plot(reduction, soh_improvement, 'bo-', label='SOH', linewidth=2, markersize=8)
ax4.plot(reduction, soc_improvement, 'ro-', label='SOC', linewidth=2, markersize=8)
ax4.set_xlabel('Feature Reduction (%)')
ax4.set_ylabel('Improvement (%)')
ax4.set_title('Benefit of Feature Reduction')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# SAVE RESULTS
# ============================================================
results_df.to_csv(os.path.join(data_dir, 'feature_selection_results_fixed.csv'), index=False)
importance_soh.to_csv(os.path.join(data_dir, 'feature_importance_soh_lgb_fixed.csv'), index=False)
importance_soc.to_csv(os.path.join(data_dir, 'feature_importance_soc_lgb_fixed.csv'), index=False)

# Save recommended features
with open(os.path.join(data_dir, 'recommended_features_fixed.txt'), 'w') as f:
    f.write("Recommended Features for Joint SOH and SOC Estimation\n")
    f.write("="*50 + "\n")
    f.write(f"Feature Set: {best_balanced['Feature Set']}\n")
    f.write(f"Number of Features: {len(best_features)}\n\n")
    f.write("Features:\n")
    for i, feat in enumerate(sorted(best_features)):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nExpected SOH MAPE: {best_balanced['SOH MAPE Avg']:.2f}%\n")
    f.write(f"Expected SOC MAPE: {best_balanced['SOC MAPE Avg']:.2f}%\n")
    f.write(f"Feature Reduction: 18 → {len(best_features)} features ({((18 - len(best_features)) / 18 * 100):.1f}% reduction)\n")

print("\nResults saved to:")
print(f"  {os.path.join(data_dir, 'feature_selection_results_fixed.csv')}")
print(f"  {os.path.join(data_dir, 'feature_importance_soh_lgb_fixed.csv')}")
print(f"  {os.path.join(data_dir, 'feature_importance_soc_lgb_fixed.csv')}")
print(f"  {os.path.join(data_dir, 'recommended_features_fixed.txt')}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("FINAL SUMMARY - FEATURE SELECTION EFFECT ANALYSIS")
print("="*80)

print(f"""
FEATURE SELECTION SUMMARY

Baseline (All 18 features):
  SOH MAPE: {baseline_soh:.2f}%
  SOC MAPE: {baseline_soc:.2f}%

Best Feature Set: {best_balanced['Feature Set']}
  Features used: {len(best_features)}
  SOH MAPE: {best_balanced['SOH MAPE Avg']:.2f}% ({(baseline_soh - best_balanced['SOH MAPE Avg']):.2f}% improvement)
  SOC MAPE: {best_balanced['SOC MAPE Avg']:.2f}% ({(baseline_soc - best_balanced['SOC MAPE Avg']):.2f}% improvement)
  SOH R²: {best_balanced['SOH R² Avg']:.4f}
  SOC R²: {best_balanced['SOC R² Avg']:.4f}

Feature Reduction: 18 → {len(best_features)} features ({((18 - len(best_features)) / 18 * 100):.1f}% reduction)

KEY INSIGHT:
  Using only the most important features reduces model complexity
  while maintaining or improving prediction accuracy.
""")

print("="*80)
print("STEP 2 COMPLETE")
print("="*80)