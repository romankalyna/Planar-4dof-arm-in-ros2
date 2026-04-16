#!/usr/bin/env python3
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")  # important for ROS terminals / no-GUI environments
import matplotlib.pyplot as plt

import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="input CSV (joint_states_log.csv or arm_log.csv)")
    ap.add_argument("--out", default="joints_vs_time.png", help="output image filename")
    ap.add_argument("--title", default="Joint angles vs time", help="plot title")
    args = ap.parse_args()

    # Load CSV with header (utf-8-sig strips BOM if present)
    data = np.genfromtxt(
        args.infile,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
    )

    # Debug: show what columns were parsed
    print("Columns:", data.dtype.names)

    t = data["t"]
    print("Samples:", len(t))
    if len(t) == 0:
        raise RuntimeError("No rows loaded from CSV (is the file empty or header not parsed?)")

    q = np.vstack([data["q1_rad"], data["q2_rad"], data["q3_rad"], data["q4_rad"]]).T

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(t, q[:, i], label=f"q{i+1}")

    plt.xlabel("time (s)")
    plt.ylabel("joint angle (rad)")
    plt.title(args.title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.abspath(args.out)
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()