import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Path to the dataset
data_dir = r'C:\Users\admin\Desktop\DR2\11 All Datasets\01 NASA PCoE Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4'

# Get all mat files
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
mat_files.sort()

print("="*80)
print("NASA BATTERY DATASET EXPLORATION REPORT")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total files: {len(mat_files)}")
print("="*80)

# Store all data for final report
all_batteries = {}
all_capacities = {}
all_impedance = {}
summary_data = []

for file_name in mat_files:
    print(f"\nProcessing: {file_name}")
    print("-"*50)
    
    file_path = os.path.join(data_dir, file_name)
    data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    
    # Get battery key
    battery_key = file_name.replace('.mat', '')
    battery_data = data[battery_key]
    
    if hasattr(battery_data, 'cycle'):
        battery = battery_data.cycle
    else:
        battery = battery_data
    
    print(f"  Total cycles: {len(battery)}")
    
    # Count cycle types
    types = {}
    for i, cycle in enumerate(battery):
        t = cycle.type
        if t not in types:
            types[t] = []
        types[t].append(i)
    
    print("  Cycle types:")
    for t, indices in types.items():
        print(f"    {t}: {len(indices)} cycles")
    
    # Get capacity data
    capacities = []
    cap_cycles = []
    for i, cycle in enumerate(battery):
        if cycle.type == 'discharge':
            if hasattr(cycle.data, 'Capacity'):
                cap = float(cycle.data.Capacity)
                capacities.append(cap)
                cap_cycles.append(i)
    
    if capacities:
        print(f"  Capacity data:")
        print(f"    Initial: {capacities[0]:.3f} Ah")
        print(f"    Final: {capacities[-1]:.3f} Ah")
        print(f"    Fade: {((capacities[0] - capacities[-1]) / capacities[0] * 100):.1f}%")
        
        # Check EOL
        eol_threshold = 1.4
        eol_cycle = None
        for i, cap in enumerate(capacities):
            if cap <= eol_threshold:
                eol_cycle = cap_cycles[i]
                break
        
        if eol_cycle is not None:
            print(f"    EOL reached at cycle: {eol_cycle}")
        else:
            print("    EOL not reached")
        
        all_capacities[file_name] = {'cycles': cap_cycles, 'capacities': capacities}
        
        # Store summary
        summary_data.append({
            'battery': file_name,
            'total_cycles': len(battery),
            'discharge_cycles': len(types.get('discharge', [])),
            'charge_cycles': len(types.get('charge', [])),
            'impedance_cycles': len(types.get('impedance', [])),
            'init_capacity': capacities[0],
            'final_capacity': capacities[-1],
            'capacity_fade_pct': ((capacities[0] - capacities[-1]) / capacities[0] * 100),
            'eol_reached': eol_cycle is not None,
            'eol_cycle': eol_cycle if eol_cycle is not None else 'N/A'
        })
    
    # Get impedance data
    imp_data = []
    for i, cycle in enumerate(battery):
        if cycle.type == 'impedance':
            if hasattr(cycle.data, 'Re') and hasattr(cycle.data, 'Rct'):
                imp_data.append({
                    'cycle': i,
                    'Re': float(cycle.data.Re),
                    'Rct': float(cycle.data.Rct)
                })
    
    if imp_data:
        print(f"  Impedance data:")
        print(f"    Samples: {len(imp_data)}")
        print(f"    Re range: {min([d['Re'] for d in imp_data]):.4f} - {max([d['Re'] for d in imp_data]):.4f} Ohm")
        print(f"    Rct range: {min([d['Rct'] for d in imp_data]):.4f} - {max([d['Rct'] for d in imp_data]):.4f} Ohm")
        all_impedance[file_name] = imp_data
    
    all_batteries[file_name] = battery

print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

# Create summary DataFrame
df_summary = pd.DataFrame(summary_data)
print("\nBattery Summary Table:")
print(df_summary.to_string(index=False))

print("\n" + "="*80)
print("CAPACITY DEGRADATION PLOTS")
print("="*80)

# Plot all capacities
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (battery_name, data) in enumerate(all_capacities.items()):
    if idx < 4:
        ax = axes[idx]
        ax.plot(data['cycles'], data['capacities'], 'b-o', linewidth=2, markersize=4)
        ax.axhline(y=1.4, color='r', linestyle='--', linewidth=1, label='EOL (1.4 Ah)')
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title(battery_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("IMPEDANCE ANALYSIS")
print("="*80)

# Plot impedance trends
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (battery_name, imp_data) in enumerate(all_impedance.items()):
    if idx < 4:
        ax = axes[idx]
        cycles = [d['cycle'] for d in imp_data]
        re = [d['Re'] for d in imp_data]
        rct = [d['Rct'] for d in imp_data]
        
        ax.plot(cycles, re, 'b-o', label='Re', linewidth=2, markersize=4)
        ax.plot(cycles, rct, 'r-o', label='Rct', linewidth=2, markersize=4)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Resistance (Ohm)')
        ax.set_title(battery_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("CAPACITY VS IMPEDANCE CORRELATION")
print("="*80)

# Plot correlation for each battery
for battery_name in all_capacities.keys():
    if battery_name in all_impedance:
        caps = all_capacities[battery_name]
        imp = all_impedance[battery_name]
        
        # Find matching cycles
        cap_cycles = caps['cycles']
        capacities = caps['capacities']
        
        # Get impedance values at discharge cycles (approximate)
        re_vals = []
        rct_vals = []
        cap_vals = []
        
        for imp_point in imp:
            imp_cycle = imp_point['cycle']
            # Find closest discharge cycle
            closest_idx = min(range(len(cap_cycles)), key=lambda i: abs(cap_cycles[i] - imp_cycle))
            if abs(cap_cycles[closest_idx] - imp_cycle) <= 5:  # Within 5 cycles
                re_vals.append(imp_point['Re'])
                rct_vals.append(imp_point['Rct'])
                cap_vals.append(capacities[closest_idx])
        
        if cap_vals:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].scatter(re_vals, cap_vals, alpha=0.6, s=30)
            axes[0].set_xlabel('Re (Ohm)')
            axes[0].set_ylabel('Capacity (Ah)')
            axes[0].set_title(f'{battery_name} - Capacity vs Re')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].scatter(rct_vals, cap_vals, alpha=0.6, s=30, color='red')
            axes[1].set_xlabel('Rct (Ohm)')
            axes[1].set_ylabel('Capacity (Ah)')
            axes[1].set_title(f'{battery_name} - Capacity vs Rct')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

print("\n" + "="*80)
print("FINAL STATISTICS")
print("="*80)

# Overall statistics
total_discharge = sum([d['discharge_cycles'] for d in summary_data])
total_charge = sum([d['charge_cycles'] for d in summary_data])
total_impedance = sum([d['impedance_cycles'] for d in summary_data])
total_cycles = sum([d['total_cycles'] for d in summary_data])

print(f"Total batteries analyzed: {len(mat_files)}")
print(f"Total cycles across all batteries: {total_cycles}")
print(f"  - Discharge cycles: {total_discharge}")
print(f"  - Charge cycles: {total_charge}")
print(f"  - Impedance cycles: {total_impedance}")

avg_fade = np.mean([d['capacity_fade_pct'] for d in summary_data])
print(f"\nAverage capacity fade: {avg_fade:.1f}%")

eol_count = sum([1 for d in summary_data if d['eol_reached']])
print(f"Batteries reaching EOL: {eol_count}/{len(mat_files)}")

if eol_count > 0:
    avg_eol_cycle = np.mean([d['eol_cycle'] for d in summary_data if d['eol_reached']])
    print(f"Average EOL cycle: {avg_eol_cycle:.0f}")

print("\n" + "="*80)
print("REPORT COMPLETE")
print("="*80)

# Save report
report_df = df_summary.copy()
report_df.to_csv(os.path.join(data_dir, 'battery_analysis_report.csv'), index=False)
print(f"\nReport saved to: {os.path.join(data_dir, 'battery_analysis_report.csv')}")