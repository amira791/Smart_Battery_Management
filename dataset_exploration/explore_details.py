import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from scipy import stats

# Path to the dataset
data_dir = r'C:\Users\admin\Desktop\DR2\11 All Datasets\01 NASA PCoE Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4'

# Get all mat files
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
mat_files.sort()

print("="*80)
print("NASA BATTERY DATASET DETAILED EXPLORATION REPORT")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total files: {len(mat_files)}")
print("="*80)

# Store all data
all_batteries = {}
all_capacities = {}
all_impedance = {}
summary_data = []
detailed_cycles = {}

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
    cycle_info = []
    for i, cycle in enumerate(battery):
        t = cycle.type
        if t not in types:
            types[t] = []
        types[t].append(i)
        cycle_info.append({
            'cycle': i,
            'type': t,
            'ambient_temp': cycle.ambient_temperature
        })
    
    print("  Cycle types:")
    for t, indices in types.items():
        print(f"    {t}: {len(indices)} cycles (indices: {indices[0]} to {indices[-1]})")
    
    # Get capacity data
    capacities = []
    cap_cycles = []
    voltage_data = []
    
    for i, cycle in enumerate(battery):
        if cycle.type == 'discharge':
            if hasattr(cycle.data, 'Capacity'):
                cap = float(cycle.data.Capacity)
                capacities.append(cap)
                cap_cycles.append(i)
                
                # Store detailed discharge data
                voltage_data.append({
                    'cycle': i,
                    'voltage': cycle.data.Voltage_measured,
                    'current': cycle.data.Current_measured,
                    'temp': cycle.data.Temperature_measured,
                    'time': cycle.data.Time,
                    'capacity': cap
                })
    
    if capacities:
        print(f"\n  Capacity Analysis:")
        print(f"    Initial (cycle {cap_cycles[0]}): {capacities[0]:.3f} Ah")
        print(f"    Final (cycle {cap_cycles[-1]}): {capacities[-1]:.3f} Ah")
        print(f"    Total fade: {((capacities[0] - capacities[-1]) / capacities[0] * 100):.1f}%")
        print(f"    Max capacity: {max(capacities):.3f} Ah")
        print(f"    Min capacity: {min(capacities):.3f} Ah")
        print(f"    Average capacity: {np.mean(capacities):.3f} Ah")
        print(f"    Std deviation: {np.std(capacities):.3f} Ah")
        
        # Calculate degradation rate
        cycles_range = np.array(cap_cycles)
        cap_array = np.array(capacities)
        slope, intercept, r_value, p_value, std_err = stats.linregress(cycles_range, cap_array)
        print(f"    Degradation rate: {abs(slope):.4f} Ah/cycle")
        print(f"    R-squared: {r_value**2:.4f}")
        
        # Check EOL
        eol_threshold = 1.4
        eol_cycle = None
        for i, cap in enumerate(capacities):
            if cap <= eol_threshold:
                eol_cycle = cap_cycles[i]
                break
        
        if eol_cycle is not None:
            print(f"    EOL reached at cycle: {eol_cycle}")
            cycles_to_eol = eol_cycle - cap_cycles[0]
            print(f"    Cycles to EOL: {cycles_to_eol}")
        else:
            print("    EOL not reached (capacity still above 1.4 Ah)")
        
        all_capacities[file_name] = {
            'cycles': cap_cycles, 
            'capacities': capacities,
            'voltage_data': voltage_data
        }
        
        # Store summary
        summary_data.append({
            'battery': file_name,
            'total_cycles': len(battery),
            'discharge_cycles': len(types.get('discharge', [])),
            'charge_cycles': len(types.get('charge', [])),
            'impedance_cycles': len(types.get('impedance', [])),
            'init_capacity': capacities[0],
            'final_capacity': capacities[-1],
            'max_capacity': max(capacities),
            'min_capacity': min(capacities),
            'capacity_fade_pct': ((capacities[0] - capacities[-1]) / capacities[0] * 100),
            'degradation_rate': abs(slope),
            'r_squared': r_value**2,
            'eol_reached': eol_cycle is not None,
            'eol_cycle': eol_cycle if eol_cycle is not None else 'N/A',
            'cycles_to_eol': cycles_to_eol if eol_cycle is not None else 'N/A'
        })
    
    # Get impedance data - handle different field names
    imp_data = []
    for i, cycle in enumerate(battery):
        if cycle.type == 'impedance':
            imp_point = {'cycle': i}
            
            # Check for Re and Rct (case sensitive)
            if hasattr(cycle.data, 'Re'):
                imp_point['Re'] = float(cycle.data.Re)
            elif hasattr(cycle.data, 're'):
                imp_point['Re'] = float(cycle.data.re)
            
            if hasattr(cycle.data, 'Rct'):
                imp_point['Rct'] = float(cycle.data.Rct)
            elif hasattr(cycle.data, 'rct'):
                imp_point['Rct'] = float(cycle.data.rct)
            
            # Check impedance fields
            if hasattr(cycle.data, 'Battery_impedance'):
                imp_point['Battery_impedance'] = cycle.data.Battery_impedance
            elif hasattr(cycle.data, 'battery_impedance'):
                imp_point['Battery_impedance'] = cycle.data.battery_impedance
            
            if hasattr(cycle.data, 'Rectified_Impedance'):
                imp_point['Rectified_impedance'] = cycle.data.Rectified_Impedance
            elif hasattr(cycle.data, 'Rectified_impedance'):
                imp_point['Rectified_impedance'] = cycle.data.Rectified_impedance
            
            if hasattr(cycle.data, 'Sense_current'):
                imp_point['Sense_current'] = cycle.data.Sense_current
            elif hasattr(cycle.data, 'sense_current'):
                imp_point['Sense_current'] = cycle.data.sense_current
            
            if hasattr(cycle.data, 'Battery_current'):
                imp_point['Battery_current'] = cycle.data.Battery_current
            elif hasattr(cycle.data, 'battery_current'):
                imp_point['Battery_current'] = cycle.data.battery_current
            
            if hasattr(cycle.data, 'Current_ratio'):
                imp_point['Current_ratio'] = cycle.data.Current_ratio
            elif hasattr(cycle.data, 'current_ratio'):
                imp_point['Current_ratio'] = cycle.data.current_ratio
            
            # Only add if we have Re and Rct
            if 'Re' in imp_point and 'Rct' in imp_point:
                imp_data.append(imp_point)
    
    if imp_data:
        print(f"\n  Impedance Analysis:")
        print(f"    Samples: {len(imp_data)}")
        re_vals = [d['Re'] for d in imp_data]
        rct_vals = [d['Rct'] for d in imp_data]
        print(f"    Re: min={min(re_vals):.4f}, max={max(re_vals):.4f}, mean={np.mean(re_vals):.4f} Ohm")
        print(f"    Rct: min={min(rct_vals):.4f}, max={max(rct_vals):.4f}, mean={np.mean(rct_vals):.4f} Ohm")
        print(f"    Re increase: {((max(re_vals)-min(re_vals))/min(re_vals)*100):.1f}%")
        print(f"    Rct increase: {((max(rct_vals)-min(rct_vals))/min(rct_vals)*100):.1f}%")
        
        all_impedance[file_name] = imp_data
    
    detailed_cycles[file_name] = cycle_info

print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

# Create summary DataFrame
df_summary = pd.DataFrame(summary_data)
print("\nBattery Summary Table:")
print(df_summary.round(4).to_string(index=False))

print("\n" + "="*80)
print("CAPACITY DEGRADATION ANALYSIS")
print("="*80)

# Create detailed capacity plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, (battery_name, data) in enumerate(all_capacities.items()):
    if idx < 4:
        ax = axes[idx]
        ax.plot(data['cycles'], data['capacities'], 'b-', linewidth=2, label='Capacity')
        ax.axhline(y=1.4, color='r', linestyle='--', linewidth=2, label='EOL (1.4 Ah)')
        
        # Add trend line
        z = np.polyfit(data['cycles'], data['capacities'], 1)
        p = np.poly1d(z)
        ax.plot(data['cycles'], p(data['cycles']), 'g--', linewidth=1, label='Trend')
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title(f'{battery_name} - Capacity Degradation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotation
        init_cap = data['capacities'][0]
        final_cap = data['capacities'][-1]
        fade = ((init_cap - final_cap) / init_cap * 100)
        ax.annotate(f'Fade: {fade:.1f}%', xy=(0.05, 0.95), xycoords='axes fraction', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("IMPEDANCE ANALYSIS")
print("="*80)

# Create impedance plots
if all_impedance:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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
            ax.set_title(f'{battery_name} - Impedance Evolution')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("No impedance data available")

print("\n" + "="*80)
print("CAPACITY VS IMPEDANCE CORRELATION")
print("="*80)

# Create correlation plots
for battery_name in all_capacities.keys():
    if battery_name in all_impedance:
        caps = all_capacities[battery_name]
        imp = all_impedance[battery_name]
        
        # Match impedance cycles with discharge cycles
        cap_cycles = caps['cycles']
        capacities = caps['capacities']
        
        re_vals = []
        rct_vals = []
        cap_vals = []
        
        for imp_point in imp:
            imp_cycle = imp_point['cycle']
            # Find closest discharge cycle
            closest_idx = min(range(len(cap_cycles)), key=lambda i: abs(cap_cycles[i] - imp_cycle))
            if abs(cap_cycles[closest_idx] - imp_cycle) <= 10:
                re_vals.append(imp_point['Re'])
                rct_vals.append(imp_point['Rct'])
                cap_vals.append(capacities[closest_idx])
        
        if cap_vals:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Capacity vs Re
            axes[0].scatter(re_vals, cap_vals, alpha=0.6, s=30)
            try:
                z = np.polyfit(re_vals, cap_vals, 1)
                p = np.poly1d(z)
                axes[0].plot(re_vals, p(re_vals), 'r--', linewidth=1)
            except:
                pass
            axes[0].set_xlabel('Re (Ohm)')
            axes[0].set_ylabel('Capacity (Ah)')
            axes[0].set_title(f'{battery_name} - Capacity vs Re')
            axes[0].grid(True, alpha=0.3)
            
            # Capacity vs Rct
            axes[1].scatter(rct_vals, cap_vals, alpha=0.6, s=30, color='red')
            try:
                z = np.polyfit(rct_vals, cap_vals, 1)
                p = np.poly1d(z)
                axes[1].plot(rct_vals, p(rct_vals), 'b--', linewidth=1)
            except:
                pass
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
std_fade = np.std([d['capacity_fade_pct'] for d in summary_data])
print(f"\nAverage capacity fade: {avg_fade:.1f}% ± {std_fade:.1f}%")

avg_rate = np.mean([d['degradation_rate'] for d in summary_data])
print(f"Average degradation rate: {avg_rate:.4f} Ah/cycle")

eol_count = sum([1 for d in summary_data if d['eol_reached']])
print(f"Batteries reaching EOL: {eol_count}/{len(mat_files)}")

if eol_count > 0:
    eol_cycles = [d['eol_cycle'] for d in summary_data if d['eol_reached']]
    avg_eol = np.mean(eol_cycles)
    print(f"Average EOL cycle: {avg_eol:.0f}")

# Calculate correlations
re_vals_all = []
rct_vals_all = []
cap_vals_all = []

for battery_name in all_capacities.keys():
    if battery_name in all_impedance:
        caps = all_capacities[battery_name]
        imp = all_impedance[battery_name]
        cap_cycles = caps['cycles']
        capacities = caps['capacities']
        
        for imp_point in imp:
            imp_cycle = imp_point['cycle']
            closest_idx = min(range(len(cap_cycles)), key=lambda i: abs(cap_cycles[i] - imp_cycle))
            if abs(cap_cycles[closest_idx] - imp_cycle) <= 10:
                re_vals_all.append(imp_point['Re'])
                rct_vals_all.append(imp_point['Rct'])
                cap_vals_all.append(capacities[closest_idx])

if cap_vals_all:
    corr_re, _ = stats.pearsonr(re_vals_all, cap_vals_all)
    corr_rct, _ = stats.pearsonr(rct_vals_all, cap_vals_all)
    print(f"\nCorrelation with capacity:")
    print(f"  Re: {corr_re:.3f}")
    print(f"  Rct: {corr_rct:.3f}")

print("\n" + "="*80)
print("REPORT COMPLETE")
print("="*80)

# Save detailed report
report_df = df_summary.copy()
report_path = os.path.join(data_dir, f'battery_detailed_report_{datetime.now().strftime("%Y%m%d")}.csv')
report_df.to_csv(report_path, index=False)
print(f"\nDetailed report saved to: {report_path}")

# Save summary statistics
stats_file = os.path.join(data_dir, f'battery_statistics_{datetime.now().strftime("%Y%m%d")}.txt')
with open(stats_file, 'w') as f:
    f.write("NASA BATTERY DATASET STATISTICS\n")
    f.write("="*50 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total batteries: {len(mat_files)}\n")
    f.write(f"Total cycles: {total_cycles}\n")
    f.write(f"  - Discharge: {total_discharge}\n")
    f.write(f"  - Charge: {total_charge}\n")
    f.write(f"  - Impedance: {total_impedance}\n\n")
    f.write(f"Average capacity fade: {avg_fade:.1f}%\n")
    f.write(f"Average degradation rate: {avg_rate:.4f} Ah/cycle\n")
    f.write(f"EOL reached: {eol_count}/{len(mat_files)}\n")
    if eol_count > 0:
        f.write(f"Average EOL cycle: {avg_eol:.0f}\n")
    if cap_vals_all:
        f.write(f"\nCorrelations with capacity:\n")
        f.write(f"  Re: {corr_re:.3f}\n")
        f.write(f"  Rct: {corr_rct:.3f}\n")

print(f"Statistics saved to: {stats_file}")
print("\n" + "="*80)