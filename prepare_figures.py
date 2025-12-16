import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================

# List of paths to CSV files to plot
# You can add as many paths as you want here.
FILE_PATHS = [
    "simulation_data_quadruped_rl_fixed_0_6_no_noise.csv",
    "simulation_data_quadruped_rl_fixed_0_6_no_noise_disabled_noise.csv",
    # "/absolute/path/to/another/file.csv",
]

ADD_DES_X = False

# ==========================================
# PLOTTING SCRIPT
# ==========================================

def plot_comparison(file_paths):
    if not file_paths:
        print("No file paths specified in FILE_PATHS.")
        return

    # Create figures
    # Figure 1: Positions (x, y, z)
    fig_pos, ax_pos = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig_pos.suptitle("Position Comparison")
    
    # Figure 2: Velocities (x, y, z)
    fig_vel, ax_vel = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig_vel.suptitle("Velocity Comparison")
    
    # Figure 3: Cost of Transport (CoT)
    fig_cot, ax_cot = plt.subplots(1, 1, figsize=(10, 6))
    fig_cot.suptitle("Cost of Transport (CoT) Comparison")

    # Set titles and labels
    ax_pos[0].set_ylabel("Pos X (m)")
    ax_pos[1].set_ylabel("Pos Y (m)")
    ax_pos[2].set_ylabel("Pos Z (m)")
    ax_pos[2].set_xlabel("Time (s)")
    
    ax_vel[0].set_ylabel("Vel X (m/s)")
    ax_vel[1].set_ylabel("Vel Y (m/s)")
    ax_vel[2].set_ylabel("Vel Z (m/s)")
    ax_vel[2].set_xlabel("Time (s)")
    
    ax_cot.set_ylabel("CoT")
    ax_cot.set_xlabel("Time (s)")

    # Iterate over files and plot data
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            # Read CSV file
            # Using sep=None and engine='python' to automatically detect separator (comma, tab, etc.)
            df = pd.read_csv(file_path, sep=None, engine='python')
            
            # Clean column names (strip whitespace)
            df.columns = df.columns.str.strip()
            
            label = os.path.basename(file_path)
            print(f"Processing: {label}")

            # Plot Positions
            if 'time' in df.columns:
                if 'pos_x' in df.columns: ax_pos[0].plot(df['time'], df['pos_x'], label=label)
                if 'pos_y' in df.columns: ax_pos[1].plot(df['time'], df['pos_y'], label=label)
                if 'pos_z' in df.columns: ax_pos[2].plot(df['time'], df['pos_z'], label=label)

                # Plot Velocities
                if 'vel_x' in df.columns: 
                    ax_vel[0].plot(df['time'], df['vel_x'], label=label)
                    # Optional: Plot desired velocity if available (dashed line)
                    if 'des_vel_x' in df.columns and ADD_DES_X:
                        ax_vel[0].plot(df['time'], df['des_vel_x'], linestyle='--', alpha=0.5, label=f"{label} (des)")
                
                if 'vel_y' in df.columns: ax_vel[1].plot(df['time'], df['vel_y'], label=label)
                if 'vel_z' in df.columns: ax_vel[2].plot(df['time'], df['vel_z'], label=label)

                # Plot CoT
                if 'cot' in df.columns: ax_cot.plot(df['time'], df['cot'], label=label)
            else:
                print(f"Error: 'time' column not found in {file_path}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # Add legends and grids
    for ax in ax_pos:
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    for ax in ax_vel:
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    ax_cot.legend()
    ax_cot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comparison(FILE_PATHS)
