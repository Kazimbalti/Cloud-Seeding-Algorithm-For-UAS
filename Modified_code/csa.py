# Cloud Seeding Algorithm (CSA) - Enhanced for Research Paper
# Original code by Spencer Faber (sfaber@dropletmeasurement.com)
# Modified to include data visualization and result output for research purposes.

import numpy as np
import pandas as pd
from datetime import datetime
import math, glob
import os
import matplotlib.pyplot as plt  # Added for plotting

# Constants and Settings (Unchanged)
cdp_bin_mid = np.concatenate((np.arange(2.5, 14.5, 1), np.arange(15, 51, 2)))
cdp_bin_up = np.concatenate((np.arange(3, 14, 1), np.arange(14, 52, 2)))
cdp_bin_low = np.concatenate((np.arange(2, 14, 1), np.arange(14, 50, 2)))

css_table_path = r'lookup_tables'  
cdp_sa = 0.253  
cdp_n_cloud_thresh = 5  
seed_score_thresh = 8  
seed_dist_thresh = 40  
ave_int = 3  
lcl_t = 0  

# Import Lookup Tables (Unchanged)
table_path = glob.glob(os.path.join(css_table_path, '*'))
tables = {}
for item in table_path:
    df_name = float(item.split('.csv')[0].split('\\')[-1])
    tables[df_name] = pd.read_csv(item, index_col=0)

tables_lcl_t_float = np.empty(0)
for key in tables.keys():
    tables_lcl_t_float = np.append(tables_lcl_t_float, key)

# Cloud Seeding Algorithm (CSA)
def csa(mip_data, pops_data, cdp_data):
    tas = mip_data[27]  
    pitch = mip_data[11]  
    wind_w = mip_data[39]  
    lat = mip_data[5]  
    long = mip_data[6]  
    time = (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    pops_n = pops_data[3]  

    # CDP Bulk Parameters Calculation (Unchanged)
    cdp_bin_c_float = np.array(cdp_data[15:45], dtype=np.float32)
    if tas > 0:
        cdp_c = np.sum(cdp_bin_c_float)
        cdp_sv_cm = (cdp_sa / 100) * (tas * 100)
        cdp_n = np.float32(round(cdp_c / cdp_sv_cm, 5))

        cdp_int_mass = np.sum(cdp_bin_c_float * 1e-12 * (np.pi / 6) * cdp_bin_mid ** 3)
        cdp_sv_m = cdp_sa / 1e6 * tas
        cdp_lwc = np.float32(round(cdp_int_mass / cdp_sv_m, 5))
    else:
        cdp_n = np.float32(0)
        cdp_lwc = np.float32(0)

    if np.sum(cdp_bin_c_float) > 0 and tas > 0:
        cdp_bin_vol = cdp_bin_c_float * cdp_bin_low ** 3
        cdp_vol_pro = cdp_bin_vol / np.sum(cdp_bin_vol)
        cdp_vol_pro_c_sum = np.cumsum(cdp_vol_pro)
        i_v50 = np.argwhere(cdp_vol_pro_c_sum > 0.5)[0][0]

        cdp_mvd = np.float32(round(cdp_bin_low[i_v50] + ((0.5 - cdp_vol_pro_c_sum[i_v50 - 1]) / cdp_vol_pro[i_v50]) * 
                                   (cdp_bin_low[i_v50 + 1] - cdp_bin_low[i_v50]), 5))
    else:
        cdp_mvd = np.float32(0)

    # Seedability Evaluation (Unchanged)
    nearest_lcl_t = (np.abs(tables_lcl_t_float - lcl_t)).argmin()
    table_df = tables[tables_lcl_t_float[nearest_lcl_t]]

    wind_w_ave, wind_w_std, pops_n_ave, pops_n_std = ave_measurements(wind_w, pops_n)

    if np.isfinite(wind_w_ave):
        x_ind = (np.abs(table_df.index - wind_w_ave)).argmin()
        y_ind = (np.abs(table_df.columns.astype(float) - pops_n_ave)).argmin()
        seed_score = table_df.iloc[x_ind, y_ind]
    else:
        seed_score = 0

    seed_switch = seed_scale(seed_score, tas, pitch)

    return [time, lat, long, cdp_n, cdp_lwc, cdp_mvd, seed_score, seed_switch]

# Seed Scale (Unchanged)
seedable_dist = 0
def seed_scale(seed_score, tas, pitch):
    global seed_score_thresh
    global seed_dist_thresh
    global seedable_dist

    if seed_score >= seed_score_thresh:
        x_dist = tas * math.cos(math.radians(pitch))
        seedable_dist = seedable_dist + x_dist
    else:
        seedable_dist = 0

    if seedable_dist >= seed_dist_thresh:
        seed_switch = 1
    else:
        seed_switch = 0

    return seed_switch

# Averaging Measurements (Unchanged)
wind_w_ave_arr = np.zeros(ave_int)  
pops_n_ave_arr = np.zeros(ave_int)  
wind_w_ave_arr[:] = np.NaN          
pops_n_ave_arr[:] = np.NaN          

def ave_measurements(wind_w, pops_n):
    global wind_w_ave_arr
    global pops_n_ave_arr

    if len(np.argwhere(np.isnan(wind_w_ave_arr))) > 0:
        nan_min_i = np.min(np.argwhere(np.isnan(wind_w_ave_arr)))
        wind_w_ave_arr[nan_min_i] = wind_w
        pops_n_ave_arr[nan_min_i] = pops_n
    else:
        wind_w_ave_arr[0] = np.NaN
        wind_w_ave_arr = np.roll(wind_w_ave_arr, -1)
        wind_w_ave_arr[-1] = wind_w

        pops_n_ave_arr[0] = np.NaN
        pops_n_ave_arr = np.roll(pops_n_ave_arr, -1)
        pops_n_ave_arr[-1] = pops_n

    i_updraft = np.where(np.nan_to_num(wind_w_ave_arr) > 0)

    if len(i_updraft[0]) > 1:
        wind_w_ave = np.mean(wind_w_ave_arr[i_updraft])
        wind_w_std = np.std(wind_w_ave_arr[i_updraft])
    else:
        wind_w_ave = np.NaN
        wind_w_std = np.NaN

    pops_n_ave = np.nanmean(pops_n_ave_arr)
    pops_n_std = np.nanstd(pops_n_ave_arr)

    return [wind_w_ave, wind_w_std, pops_n_ave, pops_n_std]
