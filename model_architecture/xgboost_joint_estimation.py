import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Path to the dataset
data_dir = r'C:\Users\admin\Desktop\DR2\11 All Datasets\01 NASA PCoE Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4'

print("="*80)
print("JOINT SOH AND SOC ESTIMATION - 18 FEATURES (BATTERY-BASED SPLIT)")
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

print(f"\nFeatures used ({len(feature_columns)} features):")
for i, feature in enumerate(feature_columns):
    print(f"  {i+1}. {feature}")

# ============================================================
# LEAVE-ONE-BATTERY-OUT CROSS-VALIDATION
# ============================================================
print("\n" + "="*80)
print("LEAVE-ONE-BATTERY-OUT CROSS-VALIDATION (18 features)")
print("="*80)

batteries = ['B0005', 'B0006', 'B0007', 'B0018']
cv_results = []

for test_battery in batteries:
    train_batteries = [b for b in batteries if b != test_battery]
    
    print(f"\n{'='*60}")
    print(f"Testing on: {test_battery}")
    print(f"Training on: {train_batteries}")
    print(f"{'='*60}")
    
    # Split data
    train_mask = df_all['battery_id'].isin(train_batteries)
    test_mask = df_all['battery_id'] == test_battery
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_soh_train = y_soh[train_mask]
    y_soh_test = y_soh[test_mask]
    y_soc_train = y_soc[train_mask]
    y_soc_test = y_soc[test_mask]
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # TRAIN SOH MODEL
    # ============================================================
    print("\nTraining SOH Model...")
    model_soh = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    model_soh.fit(X_train_scaled, y_soh_train)
    
    # ============================================================
    # TRAIN SOC MODEL
    # ============================================================
    print("Training SOC Model...")
    model_soc = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    model_soc.fit(X_train_scaled, y_soc_train)
    
    # ============================================================
    # EVALUATE
    # ============================================================
    pred_soh = model_soh.predict(X_test_scaled)
    pred_soc = model_soc.predict(X_test_scaled)
    
    # SOH Metrics
    mae_soh = mean_absolute_error(y_soh_test, pred_soh)
    rmse_soh = np.sqrt(mean_squared_error(y_soh_test, pred_soh))
    r2_soh = r2_score(y_soh_test, pred_soh)
    mape_soh = np.mean(np.abs((y_soh_test - pred_soh) / y_soh_test)) * 100
    
    # SOC Metrics
    mae_soc = mean_absolute_error(y_soc_test, pred_soc)
    rmse_soc = np.sqrt(mean_squared_error(y_soc_test, pred_soc))
    r2_soc = r2_score(y_soc_test, pred_soc)
    mape_soc = np.mean(np.abs((y_soc_test - pred_soc) / y_soc_test)) * 100
    
    cv_results.append({
        'test_battery': test_battery,
        'samples': len(X_test),
        'soh_mae': mae_soh,
        'soh_rmse': rmse_soh,
        'soh_r2': r2_soh,
        'soh_mape': mape_soh,
        'soc_mae': mae_soc,
        'soc_rmse': rmse_soc,
        'soc_r2': r2_soc,
        'soc_mape': mape_soc
    })
    
    print(f"\n{'='*40}")
    print(f"RESULTS FOR {test_battery}")
    print(f"{'='*40}")
    print(f"\nSOH Estimation:")
    print(f"  MAE:  {mae_soh:.4f}")
    print(f"  RMSE: {rmse_soh:.4f}")
    print(f"  R²:   {r2_soh:.4f}")
    print(f"  MAPE: {mape_soh:.2f}%")
    print(f"\nSOC Estimation:")
    print(f"  MAE:  {mae_soc:.4f}")
    print(f"  RMSE: {rmse_soc:.4f}")
    print(f"  R²:   {r2_soc:.4f}")
    print(f"  MAPE: {mape_soc:.2f}%")

# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================
print("\n" + "="*80)
print("CROSS-VALIDATION SUMMARY - 18 FEATURES")
print("="*80)

cv_df = pd.DataFrame(cv_results)
print("\nPer-Battery Results:")
print("="*80)
print(f"{'Test Battery':<15} {'Samples':<10} {'SOH MAPE':<12} {'SOH R²':<12} {'SOC MAPE':<12} {'SOC R²':<12}")
print("-"*80)
for _, row in cv_df.iterrows():
    print(f"{row['test_battery']:<15} {row['samples']:<10} {row['soh_mape']:<12.2f} {row['soh_r2']:<12.4f} {row['soc_mape']:<12.2f} {row['soc_r2']:<12.4f}")

avg_soh_mape = cv_df['soh_mape'].mean()
avg_soh_r2 = cv_df['soh_r2'].mean()
std_soh_mape = cv_df['soh_mape'].std()

avg_soc_mape = cv_df['soc_mape'].mean()
avg_soc_r2 = cv_df['soc_r2'].mean()
std_soc_mape = cv_df['soc_mape'].std()

print("\n" + "="*80)
print("AVERAGE PERFORMANCE - 18 FEATURES")
print("="*80)
print(f"\nSOH Estimation (18 features):")
print(f"  Average MAPE: {avg_soh_mape:.2f}% ± {std_soh_mape:.2f}%")
print(f"  Average R²:   {avg_soh_r2:.4f}")
print(f"  Best Battery: {cv_df.loc[cv_df['soh_mape'].idxmin(), 'test_battery']} ({cv_df['soh_mape'].min():.2f}%)")
print(f"  Worst Battery: {cv_df.loc[cv_df['soh_mape'].idxmax(), 'test_battery']} ({cv_df['soh_mape'].max():.2f}%)")

print(f"\nSOC Estimation (18 features):")
print(f"  Average MAPE: {avg_soc_mape:.2f}% ± {std_soc_mape:.2f}%")
print(f"  Average R²:   {avg_soc_r2:.4f}")
print(f"  Best Battery: {cv_df.loc[cv_df['soc_mape'].idxmin(), 'test_battery']} ({cv_df['soc_mape'].min():.2f}%)")
print(f"  Worst Battery: {cv_df.loc[cv_df['soc_mape'].idxmax(), 'test_battery']} ({cv_df['soc_mape'].max():.2f}%)")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Train final model on all data for feature importance
scaler_final = RobustScaler()
X_scaled_final = scaler_final.fit_transform(X)

model_soh_final = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
model_soh_final.fit(X_scaled_final, y_soh)

model_soc_final = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
model_soc_final.fit(X_scaled_final, y_soc)

importance_soh = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_soh_final.feature_importances_
}).sort_values('importance', ascending=False)

importance_soc = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_soc_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\nSOH Feature Importance (18 features):")
print("="*60)
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<15} {'Cumulative':<15}")
print("-"*60)
importance_soh['cumulative'] = importance_soh['importance'].cumsum()
for i in range(len(importance_soh)):
    row = importance_soh.iloc[i]
    print(f"{i+1:<6} {row['feature']:<25} {row['importance']:.6f}    {row['cumulative']:.6f}")

print("\nSOC Feature Importance (18 features):")
print("="*60)
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<15} {'Cumulative':<15}")
print("-"*60)
importance_soc['cumulative'] = importance_soc['importance'].cumsum()
for i in range(len(importance_soc)):
    row = importance_soc.iloc[i]
    print(f"{i+1:<6} {row['feature']:<25} {row['importance']:.6f}    {row['cumulative']:.6f}")

# ============================================================
# FEATURE SELECTION ANALYSIS
# ============================================================
print("\n" + "="*80)
print("FEATURE SELECTION ANALYSIS")
print("="*80)

for target in [0.95, 0.98, 0.99]:
    n_soh = (importance_soh['cumulative'] <= target).sum() + 1
    n_soc = (importance_soc['cumulative'] <= target).sum() + 1
    print(f"\nTo reach {target*100:.0f}% cumulative importance:")
    print(f"  SOH: {n_soh} features needed")
    print(f"  SOC: {n_soc} features needed")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SOH Feature Importance
ax1 = axes[0, 0]
ax1.barh(importance_soh['feature'], importance_soh['importance'], color='blue', alpha=0.7)
ax1.set_xlabel('Importance Score')
ax1.set_title('SOH Feature Importance (18 features)')
ax1.grid(True, alpha=0.3)

# SOH Cumulative
ax2 = axes[0, 1]
ax2.plot(range(1, len(importance_soh)+1), importance_soh['cumulative'], 'b-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
ax2.axhline(y=0.98, color='g', linestyle='--', label='98%')
ax2.axhline(y=0.99, color='orange', linestyle='--', label='99%')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Cumulative Importance')
ax2.set_title('SOH - Cumulative Feature Importance')
ax2.legend()
ax2.grid(True, alpha=0.3)

# SOC Feature Importance
ax3 = axes[1, 0]
ax3.barh(importance_soc['feature'], importance_soc['importance'], color='green', alpha=0.7)
ax3.set_xlabel('Importance Score')
ax3.set_title('SOC Feature Importance (18 features)')
ax3.grid(True, alpha=0.3)

# SOC Cumulative
ax4 = axes[1, 1]
ax4.plot(range(1, len(importance_soc)+1), importance_soc['cumulative'], 'g-', linewidth=2)
ax4.axhline(y=0.95, color='r', linestyle='--', label='95%')
ax4.axhline(y=0.98, color='g', linestyle='--', label='98%')
ax4.axhline(y=0.99, color='orange', linestyle='--', label='99%')
ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Cumulative Importance')
ax4.set_title('SOC - Cumulative Feature Importance')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# SAVE RESULTS
# ============================================================
cv_df.to_csv(os.path.join(data_dir, 'cv_results_18_features.csv'), index=False)
importance_soh.to_csv(os.path.join(data_dir, 'feature_importance_soh_18.csv'), index=False)
importance_soc.to_csv(os.path.join(data_dir, 'feature_importance_soc_18.csv'), index=False)

print("\nResults saved to:")
print(f"  {os.path.join(data_dir, 'cv_results_18_features.csv')}")
print(f"  {os.path.join(data_dir, 'feature_importance_soh_18.csv')}")
print(f"  {os.path.join(data_dir, 'feature_importance_soc_18.csv')}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("FINAL SUMMARY - 18 FEATURES (BATTERY-BASED SPLIT)")
print("="*80)

print(f"""
DATASET SUMMARY:
  - Total batteries: 4 (B0005, B0006, B0007, B0018)
  - Total discharge cycles: {len(df_all)}
  - Features used: {len(feature_columns)}
  - Cross-validation: Leave-One-Battery-Out (4 folds)

PERFORMANCE SUMMARY:
  SOH Estimation:
    - Average MAPE: {avg_soh_mape:.2f}% ± {std_soh_mape:.2f}%
    - Average R²: {avg_soh_r2:.4f}
    - Best: {cv_df.loc[cv_df['soh_mape'].idxmin(), 'test_battery']} ({cv_df['soh_mape'].min():.2f}%)
    - Worst: {cv_df.loc[cv_df['soh_mape'].idxmax(), 'test_battery']} ({cv_df['soh_mape'].max():.2f}%)
  
  SOC Estimation:
    - Average MAPE: {avg_soc_mape:.2f}% ± {std_soc_mape:.2f}%
    - Average R²: {avg_soc_r2:.4f}
    - Best: {cv_df.loc[cv_df['soc_mape'].idxmin(), 'test_battery']} ({cv_df['soc_mape'].min():.2f}%)
    - Worst: {cv_df.loc[cv_df['soc_mape'].idxmax(), 'test_battery']} ({cv_df['soc_mape'].max():.2f}%)

TOP 3 FEATURES FOR SOH:
  1. {importance_soh.iloc[0]['feature']}: {importance_soh.iloc[0]['importance']:.4f}
  2. {importance_soh.iloc[1]['feature']}: {importance_soh.iloc[1]['importance']:.4f}
  3. {importance_soh.iloc[2]['feature']}: {importance_soh.iloc[2]['importance']:.4f}

TOP 3 FEATURES FOR SOC:
  1. {importance_soc.iloc[0]['feature']}: {importance_soc.iloc[0]['importance']:.4f}
  2. {importance_soc.iloc[1]['feature']}: {importance_soc.iloc[1]['importance']:.4f}
  3. {importance_soc.iloc[2]['feature']}: {importance_soc.iloc[2]['importance']:.4f}
""")

print("="*80)
print("COMPLETED SUCCESSFULLY")
print("="*80)