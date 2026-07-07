import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Path to the dataset
data_dir = r'C:\Users\admin\Desktop\DR2\11 All Datasets\01 NASA PCoE Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4'

# Load the data
file_path = os.path.join(data_dir, 'B0005.mat')
data = loadmat(file_path, struct_as_record=False, squeeze_me=True)

# The battery data is stored differently - let's check
battery_data = data['B0005']

# Check what battery_data actually is
print("Type:", type(battery_data))
print("Has cycle attribute:", hasattr(battery_data, 'cycle'))

# If it has 'cycle', use that
if hasattr(battery_data, 'cycle'):
    battery = battery_data.cycle
else:
    # If it's directly the cycles array
    battery = battery_data

print("Total cycles:", len(battery))
print("\nAvailable cycle types:")

# Show cycle types
types = {}
for i, cycle in enumerate(battery):
    t = cycle.type
    if t not in types:
        types[t] = []
    types[t].append(i)

for t, indices in types.items():
    print(f"  {t}: {len(indices)} cycles (first at index {indices[0]})")

print("\n" + "="*60)
print("INTERACTIVE EXPLORATION")
print("="*60)
print("\nCommands:")
print("  - Enter a cycle number to see its data")
print("  - 'c' to see capacity over all cycles")
print("  - 'q' to quit")

while True:
    cmd = input("\n> ").strip()
    
    if cmd == 'q':
        break
    
    elif cmd == 'c':
        # Show capacity evolution
        cycles = []
        caps = []
        for i, cycle in enumerate(battery):
            if cycle.type == 'discharge':
                if hasattr(cycle.data, 'Capacity'):
                    cycles.append(i)
                    caps.append(cycle.data.Capacity)
        
        if caps:
            plt.figure(figsize=(10, 6))
            plt.plot(cycles, caps, 'b-o')
            plt.xlabel('Cycle Number')
            plt.ylabel('Capacity (Ah)')
            plt.title('Battery Capacity Degradation')
            plt.grid(True)
            plt.axhline(y=1.4, color='r', linestyle='--', label='EOL (1.4 Ah)')
            plt.legend()
            plt.show()
        else:
            print("No capacity data found")
    
    else:
        try:
            idx = int(cmd)
            if 0 <= idx < len(battery):
                cycle = battery[idx]
                print(f"\nCycle {idx}: {cycle.type}")
                print(f"Ambient temp: {cycle.ambient_temperature} C")
                
                d = cycle.data
                print("\nFields:")
                
                # Show what fields exist
                fields = [f for f in dir(d) if not f.startswith('_')]
                for field in fields:
                    val = getattr(d, field)
                    if hasattr(val, 'shape'):
                        print(f"  {field}: shape {val.shape}")
                        if val.size > 0 and val.size <= 10:
                            print(f"    values: {val}")
                        elif val.size > 0:
                            print(f"    first 5: {val[:5]}")
                    else:
                        print(f"  {field}: {val}")
                
                # Plot if discharge
                if cycle.type == 'discharge':
                    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
                    
                    axes[0].plot(d.Time, d.Voltage_measured)
                    axes[0].set_ylabel('Voltage (V)')
                    axes[0].grid(True)
                    
                    axes[1].plot(d.Time, d.Current_measured)
                    axes[1].set_ylabel('Current (A)')
                    axes[1].grid(True)
                    
                    axes[2].plot(d.Time, d.Temperature_measured)
                    axes[2].set_ylabel('Temperature (C)')
                    axes[2].set_xlabel('Time (s)')
                    axes[2].grid(True)
                    
                    if hasattr(d, 'Capacity'):
                        plt.suptitle(f'Discharge Cycle {idx} - Capacity: {d.Capacity:.3f} Ah')
                    else:
                        plt.suptitle(f'Discharge Cycle {idx}')
                    
                    plt.tight_layout()
                    plt.show()
                
                # Plot if charge
                elif cycle.type == 'charge':
                    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
                    
                    axes[0].plot(d.Time, d.Voltage_measured)
                    axes[0].set_ylabel('Voltage (V)')
                    axes[0].grid(True)
                    
                    axes[1].plot(d.Time, d.Current_measured)
                    axes[1].set_ylabel('Current (A)')
                    axes[1].grid(True)
                    
                    axes[2].plot(d.Time, d.Temperature_measured)
                    axes[2].set_ylabel('Temperature (C)')
                    axes[2].set_xlabel('Time (s)')
                    axes[2].grid(True)
                    
                    plt.suptitle(f'Charge Cycle {idx}')
                    plt.tight_layout()
                    plt.show()
                
                # Plot if impedance
                elif cycle.type == 'impedance':
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Nyquist plot
                    z = d.Rectified_impedance
                    axes[0].plot(np.real(z), -np.imag(z), 'b-')
                    axes[0].set_xlabel('Real Z (Ohm)')
                    axes[0].set_ylabel('-Imag Z (Ohm)')
                    axes[0].set_title('Nyquist Plot')
                    axes[0].grid(True)
                    
                    # Impedance vs frequency
                    freqs = np.logspace(-1, 3.7, len(z))
                    axes[1].loglog(freqs, np.abs(z), 'r-')
                    axes[1].set_xlabel('Frequency (Hz)')
                    axes[1].set_ylabel('|Z| (Ohm)')
                    axes[1].set_title('Impedance Spectrum')
                    axes[1].grid(True)
                    
                    plt.suptitle(f'Impedance Cycle {idx} - Re: {d.Re:.4f}, Rct: {d.Rct:.4f}')
                    plt.tight_layout()
                    plt.show()
            else:
                print(f"Cycle {idx} not found (0-{len(battery)-1})")
        except ValueError:
            print("Invalid command. Enter a number, 'c', or 'q'")