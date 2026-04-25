import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="training_no2_labels.csv")
parser.add_argument("--output", default="no2_10min.csv")
parser.add_argument("--ts", choices=["iso", "unix"], default="iso")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input, header=None, names=["timestamp", "no2"], skiprows=1)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"[INFO] Data loaded: {len(df)} hourly rows")
print(f"       Range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")

# Cubic spline
# Robust conversion: timedelta avoids depending on internal dtype (us vs ns)
t_epoch  = (df["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
no2_vals = df["no2"].values
cs = CubicSpline(t_epoch, no2_vals, bc_type="not-a-knot")

# 10-minute grid
t_start = int(t_epoch.iloc[0])
t_end   = int(t_epoch.iloc[-1])
step    = 10 * 60
t_fine  = np.arange(t_start, t_end + step, step)
no2_fine = cs(t_fine)
no2_fine = np.clip(no2_fine, 0, None)

# Output DataFrame
timestamps_dt = pd.to_datetime(t_fine, unit="s", utc=False)
if args.ts == "unix":
    ts_col  = t_fine.astype(int)
    ts_name = "timestamp_unix"
else:
    ts_col  = timestamps_dt.strftime("%Y-%m-%d %H:%M:%S")
    ts_name = "timestamp"

out = pd.DataFrame({ts_name: ts_col, "no2": np.round(no2_fine, 2)})
out.to_csv(args.output, index=False)
print(f"[OK]   Saved '{args.output}' with {len(out)} rows at 10-minute resolution")

print("\n── First 7 output rows ──")
print(out.head(7).to_string(index=False))
print("\n── Interpolated NO2 statistics ──")
print(out["no2"].describe().round(2))
