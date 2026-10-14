# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:33:26 2026

@author: 10798
"""

# -*- coding: utf-8 -*-
"""
simulation.py（并行版本 - 实时写入CSV + 断点续跑）
===================================================
每完成一次 simulation 立刻追加写入 CSV，防止断点丢失进度。
重新启动时自动跳过已完成的 sim_id。
"""

# -*- coding: utf-8 -*-
"""
simulation.py（并行版本 - 内存优化 + 实时写入CSV + 断点续跑）
=============================================================
内存优化：6个进程改为3个进程分两批，避免内存溢出。
实时写入：每完成一次 simulation 立刻追加写入 CSV。
断点续跑：重启时自动跳过已完成的 sim_id。
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import importlib
import time
import os

# =============================================================================
# 1. 读取 Population 数据
# =============================================================================

def load_population():
    prop_data     = np.genfromtxt('simdata_0.3.csv',     delimiter=",", skip_header=1)
    non_prop_data = np.genfromtxt('simdata_0.3.csv', delimiter=",", skip_header=1)

    y     = non_prop_data[:, 0]
    d_b   = prop_data[:, 5]
    n_p   = np.delete(non_prop_data, [0,  5], 1)
    o_p   = np.delete(prop_data,     [0,  5], 1)
    x_bar = n_p.mean(axis=0)
    N_pop = n_p.shape[0]

    return {
        "y":     y,
        "d_b":   d_b,
        "n_p":   n_p,
        "o_p":   o_p,
        "x_bar": x_bar,
        "N_pop": N_pop,
    }

# =============================================================================
# 2. 断点续跑：检查 CSV 中已完成的 sim_id
# =============================================================================

def get_completed_sims(csv_name):
    if not os.path.exists(csv_name):
        return set()
    try:
        df = pd.read_csv(csv_name)
        if df.empty or "sim_id" not in df.columns:
            return set()
        return set(df["sim_id"].astype(int).tolist())
    except Exception:
        return set()

# =============================================================================
# 3. 单个方法的完整运行函数
# =============================================================================

def run_method(args):
    method_name, pop_data, N_SIM = args

    module   = importlib.import_module(method_name)
    csv_name = f"{method_name}_results.csv"
    pid      = os.getpid()

    completed = get_completed_sims(csv_name)
    if completed:
        print(f"[PID {pid}] {method_name}: resuming, "
              f"{len(completed)}/{N_SIM} done, "
              f"continuing from sim_id {max(completed)+1}", flush=True)
    else:
        pd.DataFrame(columns=[
            "sim_id", "posterior_mean", "posterior_std",
            "hpd_lower", "hpd_upper"
        ]).to_csv(csv_name, index=False)
        print(f"[PID {pid}] START {method_name}", flush=True)

    t0 = time.time()

    for sim_id in range(1, N_SIM + 1):
        if sim_id in completed:
            continue

        t_sim = time.time()
        stats = module.run(pop_data, sim_id)
        elapsed = time.time() - t_sim

        row = pd.DataFrame([{
            "sim_id":         sim_id,
            "posterior_mean": stats["posterior_mean"],
            "posterior_std":  stats["posterior_std"],
            "hpd_lower":      stats["hpd_lower"],
            "hpd_upper":      stats["hpd_upper"],
        }])
        row.to_csv(csv_name, mode='a', header=False, index=False)

        print(
            f"[PID {pid}] {method_name} sim {sim_id:3d}/{N_SIM} "
            f"mean={stats['posterior_mean']:.4f} "
            f"std={stats['posterior_std']:.4f} "
            f"HPD=({stats['hpd_lower']:.4f}, {stats['hpd_upper']:.4f}) "
            f"[{elapsed:.1f}s]",
            flush=True
        )

    total = time.time() - t0
    print(f"[PID {pid}] DONE  {method_name}  total={total/3600:.2f}h", flush=True)

# =============================================================================
# 4. 主程序：分两批并行，每批3个方法，控制内存用量
# =============================================================================

if __name__ == "__main__":

    N_SIM      = 100
    N_WORKERS  = 3     # 每批3个进程，避免16GB内存溢出

    print("Loading population data...", flush=True)
    pop_data = load_population()
    print(f"Population size: {pop_data['N_pop']}\n", flush=True)

    # 分两批：较轻的方法先跑，较重的方法后跑
    # method_3/4 无混合模型最轻，method_1/2 中等，method_5/6 两阶段最重
    BATCH_1 = ["method_3", "method_4", "method_6"]
    BATCH_2 = ["method_1", "method_2", "method_5"]

    t_start = time.time()

    for batch_num, batch in enumerate([BATCH_1, BATCH_2], start=1):
        print(f"\n{'='*60}", flush=True)
        print(f"Batch {batch_num}/2: {batch}", flush=True)
        print(f"{'='*60}\n", flush=True)

        args_list = [(name, pop_data, N_SIM) for name in batch]

        with mp.Pool(processes=N_WORKERS) as pool:
            pool.map(run_method, args_list)

        print(f"\nBatch {batch_num} completed.", flush=True)

    t_total = time.time() - t_start
    print(f"\nAll simulations completed. Total time: {t_total/3600:.2f}h", flush=True)