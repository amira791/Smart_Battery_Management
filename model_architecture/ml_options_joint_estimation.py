import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Path to the dataset
data_dir = r'C:\Users\admin\Desktop\DR2\11 All Datasets\01 NASA PCoE Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4'

print("="*80)
print("JOINT SOH AND SOC ESTIMATION - ARCHITECTURE COMPARISON (FIXED)")
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

print(f"\nFeatures used ({len(feature_columns)} features):")
for i, feature in enumerate(feature_columns):
    print(f"  {i+1}. {feature}")

print("\n" + "="*80)
print("ARCHITECTURE COMPARISON - LEAVE-ONE-BATTERY-OUT CV (FIXED)")
print("="*80)

# ============================================================
# DEFINE ARCHITECTURES WITH FIXED PARAMETERS
# ============================================================

architectures = {
    'XGBoost': {
        'soh': xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, objective='reg:squarederror'
        ),
        'soc': xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, objective='reg:squarederror'
        )
    },
    'LightGBM (Fixed)': {
        'soh': lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=5,  # FIX: Minimum data in leaf
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,        # FIX: L1 regularization
            reg_lambda=0.1,       # FIX: L2 regularization
            min_gain_to_split=0.01,  # FIX: Minimum gain to split
            random_state=42,
            verbose=-1            # FIX: Suppress warnings
        ),
        'soc': lgb.LGBMRegressor(
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
    },
    'RandomForest': {
        'soh': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42
        ),
        'soc': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42
        )
    },
    'GradientBoosting': {
        'soh': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            min_samples_split=5, random_state=42
        ),
        'soc': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            min_samples_split=5, random_state=42
        )
    },
    'Ensemble (Voting)': {
        'soh': VotingRegressor([
            ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, 
                                      num_leaves=31, min_child_samples=5, verbose=-1, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
        ]),
        'soc': VotingRegressor([
            ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                                      num_leaves=31, min_child_samples=5, verbose=-1, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
        ])
    }
}

# ============================================================
# COMPARE ARCHITECTURES
# ============================================================

batteries = ['B0005', 'B0006', 'B0007', 'B0018']
all_results = {}

for arch_name, models in architectures.items():
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE: {arch_name}")
    print(f"{'='*60}")
    
    cv_results = []
    
    for test_battery in batteries:
        train_batteries = [b for b in batteries if b != test_battery]
        
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
        
        # Train models
        model_soh = models['soh']
        model_soc = models['soc']
        
        model_soh.fit(X_train_scaled, y_soh_train)
        model_soc.fit(X_train_scaled, y_soc_train)
        
        # Predict
        pred_soh = model_soh.predict(X_test_scaled)
        pred_soc = model_soc.predict(X_test_scaled)
        
        # Calculate metrics
        mape_soh = np.mean(np.abs((y_soh_test - pred_soh) / y_soh_test)) * 100
        mape_soc = np.mean(np.abs((y_soc_test - pred_soc) / y_soc_test)) * 100
        r2_soh = r2_score(y_soh_test, pred_soh)
        r2_soc = r2_score(y_soc_test, pred_soc)
        
        cv_results.append({
            'test_battery': test_battery,
            'soh_mape': mape_soh,
            'soc_mape': mape_soc,
            'soh_r2': r2_soh,
            'soc_r2': r2_soc
        })
    
    # Store results
    all_results[arch_name] = pd.DataFrame(cv_results)
    
    # Print summary for this architecture
    print(f"\nResults for {arch_name}:")
    print("-"*60)
    print(f"{'Test Battery':<15} {'SOH MAPE':<12} {'SOH R²':<12} {'SOC MAPE':<12} {'SOC R²':<12}")
    print("-"*60)
    for _, row in all_results[arch_name].iterrows():
        print(f"{row['test_battery']:<15} {row['soh_mape']:<12.2f} {row['soh_r2']:<12.4f} {row['soc_mape']:<12.2f} {row['soc_r2']:<12.4f}")
    
    avg_soh = all_results[arch_name]['soh_mape'].mean()
    avg_soc = all_results[arch_name]['soc_mape'].mean()
    print("-"*60)
    print(f"{'AVERAGE':<15} {avg_soh:<12.2f} {'':<12} {avg_soc:<12.2f}")

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "="*80)
print("ARCHITECTURE COMPARISON SUMMARY")
print("="*80)

comparison_data = []
for arch_name, results in all_results.items():
    comparison_data.append({
        'Architecture': arch_name,
        'SOH MAPE Avg': results['soh_mape'].mean(),
        'SOH MAPE Std': results['soh_mape'].std(),
        'SOC MAPE Avg': results['soc_mape'].mean(),
        'SOC MAPE Std': results['soc_mape'].std(),
        'SOH R² Avg': results['soh_r2'].mean(),
        'SOC R² Avg': results['soc_r2'].mean()
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('SOH MAPE Avg')

print("\nComparison Table (sorted by SOH MAPE):")
print("="*100)
print(f"{'Architecture':<20} {'SOH MAPE':<15} {'SOH R²':<12} {'SOC MAPE':<15} {'SOC R²':<12}")
print("-"*100)
for _, row in comparison_df.iterrows():
    print(f"{row['Architecture']:<20} {row['SOH MAPE Avg']:<15.2f} {row['SOH R² Avg']:<12.4f} {row['SOC MAPE Avg']:<15.2f} {row['SOC R² Avg']:<12.4f}")

# ============================================================
# FIND BEST ARCHITECTURE
# ============================================================
best_soh = comparison_df.loc[comparison_df['SOH MAPE Avg'].idxmin()]
best_soc = comparison_df.loc[comparison_df['SOC MAPE Avg'].idxmin()]

print("\n" + "="*80)
print("BEST ARCHITECTURE")
print("="*80)
print(f"\nBest for SOH: {best_soh['Architecture']} ({best_soh['SOH MAPE Avg']:.2f}% MAPE)")
print(f"Best for SOC: {best_soc['Architecture']} ({best_soc['SOC MAPE Avg']:.2f}% MAPE)")

# ============================================================
# RECOMMENDATION
# ============================================================
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find the architecture with best balance
comparison_df['Score'] = (comparison_df['SOH MAPE Avg'] / comparison_df['SOH MAPE Avg'].min() + 
                          comparison_df['SOC MAPE Avg'] / comparison_df['SOC MAPE Avg'].min()) / 2
best_overall = comparison_df.loc[comparison_df['Score'].idxmin()]

print(f"\nRecommended Architecture: {best_overall['Architecture']}")
print(f"  SOH MAPE: {best_overall['SOH MAPE Avg']:.2f}%")
print(f"  SOC MAPE: {best_overall['SOC MAPE Avg']:.2f}%")
print(f"  SOH R²: {best_overall['SOH R² Avg']:.4f}")
print(f"  SOC R²: {best_overall['SOC R² Avg']:.4f}")

# ============================================================
# EXPLANATION OF FIXES
# ============================================================
print("\n" + "="*80)
print("EXPLANATION OF LIGHTGBM FIXES")
print("="*80)
print("""
The warnings 'No further splits with positive gain, best gain: -inf' were fixed by:

1. min_child_samples=5 : Ensures each leaf has at least 5 samples
2. reg_alpha=0.1      : L1 regularization (prevents overfitting)
3. reg_lambda=0.1     : L2 regularization (smooths predictions)
4. min_gain_to_split=0.01 : Minimum gain required to make a split
5. verbose=-1         : Suppresses warning messages

These parameters help LightGBM handle small datasets better
by preventing it from trying to split on noisy patterns.
""")

print("\n" + "="*80)
print("ARCHITECTURE COMPARISON COMPLETE")
print("="*80)