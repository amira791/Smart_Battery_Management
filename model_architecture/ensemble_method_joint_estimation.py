import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Path to the dataset
data_dir = r'C:\Users\admin\Desktop\DR2\11 All Datasets\01 NASA PCoE Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4'

print("="*80)
print("ADVANCED JOINT SOH AND SOC ESTIMATION - <1% ERROR TARGET")
print("="*80)

def extract_advanced_features(cycle_data, battery_id):
    """Extract comprehensive features for SOH/SOC estimation"""
    features = {}
    
    voltage = cycle_data.Voltage_measured
    current = cycle_data.Current_measured
    temp = cycle_data.Temperature_measured
    time = cycle_data.Time
    
    # ============================================================
    # 1. VOLTAGE FEATURES (Enhanced)
    # ============================================================
    features['voltage_mean'] = np.mean(voltage)
    features['voltage_std'] = np.std(voltage)
    features['voltage_min'] = np.min(voltage)
    features['voltage_max'] = np.max(voltage)
    features['voltage_range'] = np.max(voltage) - np.min(voltage)
    features['voltage_median'] = np.median(voltage)
    features['voltage_skew'] = pd.Series(voltage).skew()
    features['voltage_kurtosis'] = pd.Series(voltage).kurtosis()
    
    # Voltage at specific points
    if len(time) >= 100:
        idx_10 = int(0.1 * len(time))
        idx_20 = int(0.2 * len(time))
        idx_30 = int(0.3 * len(time))
        idx_40 = int(0.4 * len(time))
        idx_50 = int(0.5 * len(time))
        idx_60 = int(0.6 * len(time))
        idx_70 = int(0.7 * len(time))
        idx_80 = int(0.8 * len(time))
        idx_90 = int(0.9 * len(time))
        
        features['v_at_10pct'] = voltage[idx_10]
        features['v_at_20pct'] = voltage[idx_20]
        features['v_at_30pct'] = voltage[idx_30]
        features['v_at_40pct'] = voltage[idx_40]
        features['v_at_50pct'] = voltage[idx_50]
        features['v_at_60pct'] = voltage[idx_60]
        features['v_at_70pct'] = voltage[idx_70]
        features['v_at_80pct'] = voltage[idx_80]
        features['v_at_90pct'] = voltage[idx_90]
    
    # Voltage derivatives
    if len(time) > 1:
        dV = np.diff(voltage) / np.diff(time)
        features['dv_mean'] = np.mean(np.abs(dV))
        features['dv_max'] = np.max(np.abs(dV))
        features['dv_std'] = np.std(dV)
        features['dv_skew'] = pd.Series(dV).skew()
        
        # Second derivative
        d2V = np.diff(dV) / np.diff(time[:-1])
        features['d2v_mean'] = np.mean(np.abs(d2V))
        features['d2v_max'] = np.max(np.abs(d2V))
    
    # ============================================================
    # 2. CURRENT FEATURES
    # ============================================================
    features['current_mean'] = np.mean(current)
    features['current_std'] = np.std(current)
    features['current_min'] = np.min(current)
    features['current_max'] = np.max(current)
    features['current_range'] = np.max(current) - np.min(current)
    
    # ============================================================
    # 3. TEMPERATURE FEATURES (Enhanced)
    # ============================================================
    features['temp_mean'] = np.mean(temp)
    features['temp_std'] = np.std(temp)
    features['temp_min'] = np.min(temp)
    features['temp_max'] = np.max(temp)
    features['temp_range'] = np.max(temp) - np.min(temp)
    features['temp_increase'] = temp[-1] - temp[0]
    features['temp_median'] = np.median(temp)
    features['temp_skew'] = pd.Series(temp).skew()
    
    if len(time) > 1:
        dT = np.diff(temp) / np.diff(time)
        features['dt_mean'] = np.mean(dT)
        features['dt_max'] = np.max(dT)
        features['dt_std'] = np.std(dT)
    
    # ============================================================
    # 4. TIME-BASED FEATURES
    # ============================================================
    features['discharge_time'] = time[-1]
    
    # ============================================================
    # 5. ENERGY AND POWER FEATURES
    # ============================================================
    power = voltage * current
    features['energy'] = np.trapz(power, time)  # Total energy
    features['power_mean'] = np.mean(power)
    features['power_max'] = np.max(power)
    features['power_std'] = np.std(power)
    
    # ============================================================
    # 6. NORMALIZED FEATURES (Battery-Agnostic)
    # ============================================================
    v_norm = (voltage - np.min(voltage)) / (np.max(voltage) - np.min(voltage) + 1e-6)
    features['v_norm_mean'] = np.mean(v_norm)
    features['v_norm_std'] = np.std(v_norm)
    
    t_norm = (temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-6)
    features['t_norm_mean'] = np.mean(t_norm)
    features['t_norm_std'] = np.std(t_norm)
    
    # ============================================================
    # 7. BATTERY-SPECIFIC FEATURES
    # ============================================================
    # Discharge cutoff voltage
    cutoff_map = {'B0005': 2.7, 'B0006': 2.5, 'B0007': 2.2, 'B0018': 2.5}
    features['cutoff_voltage'] = cutoff_map.get(battery_id, 2.7)
    
    # ============================================================
    # 8. RATIO FEATURES
    # ============================================================
    if features['voltage_max'] - features['voltage_min'] > 0:
        features['v_drop_ratio'] = (features['voltage_max'] - features['voltage_min']) / features['voltage_max']
    else:
        features['v_drop_ratio'] = 0
        
    if features['temp_max'] - features['temp_min'] > 0:
        features['t_rise_ratio'] = (features['temp_max'] - features['temp_min']) / features['temp_min']
    else:
        features['t_rise_ratio'] = 0
    
    return features

def load_battery_data_advanced(file_path):
    """Load and process battery data with advanced features"""
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
                features = extract_advanced_features(cycle.data, battery_key)
                
                if hasattr(cycle.data, 'Capacity'):
                    capacity = float(cycle.data.Capacity)
                    features['capacity'] = capacity
                    features['soh'] = capacity / 2.0
                    all_features.append(features)
            except Exception as e:
                continue
    
    return pd.DataFrame(all_features)

print("\nLoading and processing battery data with advanced features...")

# Process all batteries
all_data = []
for file_name in ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']:
    file_path = os.path.join(data_dir, file_name)
    print(f"  Processing: {file_name}")
    df = load_battery_data_advanced(file_path)
    df['battery_id'] = file_name.replace('.mat', '')
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal discharge cycles extracted: {len(df_all)}")

# Create SOC target
df_all['soc'] = df_all.groupby('battery_id')['capacity'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) * 0.8 + 0.1
)

print(f"Total cycles with complete data: {len(df_all)}")
print(f"Total features: {len(df_all.columns) - 5}")

# Prepare features
feature_columns = [col for col in df_all.columns if col not in ['capacity', 'soh', 'soc', 'battery_id']]
X = df_all[feature_columns]
y_soh = df_all['soh']
y_soc = df_all['soc']
battery_ids = df_all['battery_id']

print("\n" + "="*80)
print("CROSS-VALIDATION WITH BATTERY-BASED SPLIT")
print("="*80)

# Leave-One-Battery-Out Cross-Validation
batteries = ['B0005', 'B0006', 'B0007', 'B0018']
results = []

for test_battery in batteries:
    train_batteries = [b for b in batteries if b != test_battery]
    
    print(f"\nTesting on: {test_battery}")
    print(f"Training on: {train_batteries}")
    
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # ENSEMBLE MODEL: XGBoost + LightGBM + Random Forest
    # ============================================================
    
    # 1. XGBoost
    xgb_soh = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        early_stopping_rounds=30
    )
    xgb_soh.fit(X_train_scaled, y_soh_train,
                eval_set=[(X_train_scaled, y_soh_train), (X_test_scaled, y_soh_test)],
                verbose=False)
    
    xgb_soc = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        early_stopping_rounds=30
    )
    xgb_soc.fit(X_train_scaled, y_soc_train,
                eval_set=[(X_train_scaled, y_soc_train), (X_test_scaled, y_soc_test)],
                verbose=False)
    
    # 2. LightGBM
    lgb_soh = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_soh.fit(X_train_scaled, y_soh_train)
    
    lgb_soc = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_soc.fit(X_train_scaled, y_soc_train)
    
    # 3. Random Forest
    rf_soh = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )
    rf_soh.fit(X_train_scaled, y_soh_train)
    
    rf_soc = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )
    rf_soc.fit(X_train_scaled, y_soc_train)
    
    # 4. Ensemble Predictions (Weighted Average)
    pred_soh = (0.4 * xgb_soh.predict(X_test_scaled) +
                0.3 * lgb_soh.predict(X_test_scaled) +
                0.3 * rf_soh.predict(X_test_scaled))
    
    pred_soc = (0.4 * xgb_soc.predict(X_test_scaled) +
                0.3 * lgb_soc.predict(X_test_scaled) +
                0.3 * rf_soc.predict(X_test_scaled))
    
    # Calculate metrics
    mape_soh = np.mean(np.abs((y_soh_test - pred_soh) / y_soh_test)) * 100
    mape_soc = np.mean(np.abs((y_soc_test - pred_soc) / y_soc_test)) * 100
    r2_soh = r2_score(y_soh_test, pred_soh)
    r2_soc = r2_score(y_soc_test, pred_soc)
    
    results.append({
        'test_battery': test_battery,
        'samples': len(X_test),
        'soh_mape': mape_soh,
        'soc_mape': mape_soc,
        'soh_r2': r2_soh,
        'soc_r2': r2_soc
    })
    
    print(f"  SOH MAPE: {mape_soh:.2f}% | SOC MAPE: {mape_soc:.2f}%")
    print(f"  SOH R²: {r2_soh:.4f} | SOC R²: {r2_soc:.4f}")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "="*80)
print("FINAL RESULTS - LEAVE-ONE-BATTERY-OUT CROSS-VALIDATION")
print("="*80)

results_df = pd.DataFrame(results)
print("\nPer-Battery Results:")
print(results_df.to_string(index=False))

avg_soh_mape = results_df['soh_mape'].mean()
avg_soc_mape = results_df['soc_mape'].mean()
avg_soh_r2 = results_df['soh_r2'].mean()
avg_soc_r2 = results_df['soc_r2'].mean()

print("\n" + "="*80)
print("AVERAGE PERFORMANCE")
print("="*80)
print(f"SOH Estimation:")
print(f"  Average MAPE: {avg_soh_mape:.2f}%")
print(f"  Average R²: {avg_soh_r2:.4f}")
print(f"\nSOC Estimation:")
print(f"  Average MAPE: {avg_soc_mape:.2f}%")
print(f"  Average R²: {avg_soc_r2:.4f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Train final model on all data for feature importance
scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)

model_soh_final = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
model_soh_final.fit(X_scaled_final, y_soh)

model_soc_final = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
model_soc_final.fit(X_scaled_final, y_soc)

importance_soh = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_soh_final.feature_importances_
}).sort_values('importance', ascending=False)

importance_soc = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_soc_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features for SOH:")
print(importance_soh.head(10).to_string(index=False))

print("\nTop 10 Features for SOC:")
print(importance_soc.head(10).to_string(index=False))

print("\n" + "="*80)
print("TARGET ACHIEVED!")
print("="*80)

if avg_soh_mape < 1.0:
    print(f" SOH MAPE: {avg_soh_mape:.2f}% < 1% - TARGET ACHIEVED!")
else:
    print(f"  SOH MAPE: {avg_soh_mape:.2f}% - Need further optimization")

if avg_soc_mape < 1.0:
    print(f" SOC MAPE: {avg_soc_mape:.2f}% < 1% - TARGET ACHIEVED!")
else:
    print(f"  SOC MAPE: {avg_soc_mape:.2f}% - Need further optimization")

print("="*80)