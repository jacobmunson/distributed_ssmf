# ssmf_mpi_2d_dist_init.py 
"""
Shifting Seasonal Matrix Factorisation - 2D Distributed Version

Implements a 2D parallelization of the SSMF algorithm using MPI, OpenMP, Numba.
The process grid partitions both matrix dimensions (d0, d1) to distribute the U and V factor matrices.

To run:
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
time mpirun -n 10 python ssmf_mpi_2d_dist_init.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet --periodicity 24 --max_regimes 20
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import argparse
import time
import warnings
from collections import defaultdict
from collections import deque 
from typing import Dict, List, Tuple
from scipy.stats import norm
import numpy as np
import pandas as pd
from mpi4py import MPI
import numba
import utils
#from ncp_distributed import ncp_distributed, mttkrp_u, mttkrp_v, mttkrp_w
#import ncp


###  Helper Functions ###
def normalize_factors_2d(U_local, V_local, W, comm_col, comm_row, eps=1e-9):
    """Normalizes distributed U and V factors and absorbs norms into W."""
    # Normalize U
    u_norms_sq_local = np.sum(U_local**2, axis=0)
    u_norms_sq = np.empty_like(u_norms_sq_local)
    comm_col.Allreduce(u_norms_sq_local, u_norms_sq, op=MPI.SUM)
    u_norms = np.sqrt(u_norms_sq)
    u_norms[u_norms < eps] = 1.0
    U_local_norm = U_local / u_norms
    
    # Normalize V
    v_norms_sq_local = np.sum(V_local**2, axis=0)
    v_norms_sq = np.empty_like(v_norms_sq_local)
    comm_row.Allreduce(v_norms_sq_local, v_norms_sq, op=MPI.SUM)
    v_norms = np.sqrt(v_norms_sq)
    v_norms[v_norms < eps] = 1.0
    V_local_norm = V_local / v_norms
    
    # Absorb weights into W
    W_norm = W * u_norms * v_norms
    
    return U_local_norm, V_local_norm, W_norm

def ncp_distributed_2d(
    r_locs, c_locs, t_globs, vals,                    # Local coordinate arrays
    U_local_init, V_local_init, W_init,               # Local initial factor partitions
    k, maxit, comm_grid, comm_row, comm_col, eps=1e-9 # Other parameters

):
    """2D distributed NCP algorithm."""
    U_local = U_local_init.copy()
    V_local = V_local_init.copy()
    W = W_init.copy()
    
    for iteration in range(maxit):
        # Update U 
        local_vtv = V_local.T @ V_local                 # repeated pattern, gram U_i^T U_i for process i
        vtv = np.empty_like(local_vtv)                  # make empty home for full U^T U
        comm_row.Allreduce(local_vtv, vtv, op=MPI.SUM)  # collect U^T U = \sum U_i^T U_i
        wtw = W.T @ W
        Bsq_u = wtw * vtv
        MB_u_local = mttkrp_u(r_locs, c_locs, t_globs, vals, V_local, W, U_local.shape)
        MB_u_global = np.empty_like(MB_u_local)
        comm_row.Allreduce(MB_u_local, MB_u_global, op=MPI.SUM)
        L_u = np.linalg.norm(Bsq_u) + eps
        U_local = np.maximum(0, U_local - (U_local @ Bsq_u - MB_u_global) / L_u).clip(min=eps)

        # Update V 
        local_utu = U_local.T @ U_local
        utu = np.empty_like(local_utu)
        comm_col.Allreduce(local_utu, utu, op=MPI.SUM)
        Bsq_v = wtw * utu
        MB_v_local = mttkrp_v(r_locs, c_locs, t_globs, vals, U_local, W, V_local.shape)
        MB_v_global = np.empty_like(MB_v_local)
        comm_col.Allreduce(MB_v_local, MB_v_global, op=MPI.SUM)
        L_v = np.linalg.norm(Bsq_v) + eps
        V_local = np.maximum(0, V_local - (V_local @ Bsq_v - MB_v_global) / L_v).clip(min=eps)

        # Update W 
        # Need full utu and vtv
        Bsq_w = vtv * utu
        MB_w_local = mttkrp_w(r_locs, c_locs, t_globs, vals, U_local, V_local, W.shape)
        MB_w_global = np.empty_like(W)
        comm_grid.Allreduce(MB_w_local, MB_w_global, op=MPI.SUM)
        L_w = np.linalg.norm(Bsq_w) + eps
        W = np.maximum(0, W - (W @ Bsq_w - MB_w_global) / L_w).clip(min=eps)

    return normalize_factors_2d(U_local, V_local, W, comm_col, comm_row)

def get_partition_info(dim: int, num_partitions: int) -> Tuple[List[int], List[int]]:
    """Calculates counts and displacements for a dimension partition."""
    base, extra = dim // num_partitions, dim % num_partitions
    counts = [base + 1 if i < extra else base for i in range(num_partitions)]
    displs = np.insert(np.cumsum(counts), 0, 0)[:-1].tolist()

    return counts, displs


# Numba Helper for Hybrid Parallelism

@numba.njit(parallel=True, cache=True)
def _calculate_partial_gradients_numba(U_local, V_local, D, r_locs, c_locs, vals):
    """
    A Numba-jitted, thread-parallel helper function to compute the initial
    gradient from the sparse triplet data.
    """
    # Create zero-filled arrays to store the results from each thread
    grad_U_p = np.zeros_like(U_local)
    grad_V_p = np.zeros_like(V_local)
    
    # Numba will automatically parallelize this loop across the threads
    # specified by the OMP_NUM_THREADS environment variable.
    for i in numba.prange(len(vals)):
        r, c, val = r_locs[i], c_locs[i], vals[i]
        
        # These operations are thread-safe: multiple threads might read from
        # the same row of V_local, they will only ever write to a unique
        # row of grad_U_p or grad_V_p, avoiding race conditions.
        # Assumes (r,c) pairs are unique, which they should be. 
        grad_U_p[r, :] += val * (V_local[c, :] @ D)
        grad_V_p[c, :] += val * (U_local[r, :] @ D)
        
    return grad_U_p, grad_V_p

# SSMF_MPI_2D class 
class SSMF_MPI_2D:
    def __init__(self, comm, comm_grid, comm_row, comm_col, local_coord_data, args):
        # MPI setup
        self.comm = comm
        self.comm_grid = comm_grid
        self.comm_row = comm_row
        self.comm_col = comm_col
        self.rank = comm.Get_rank()
        self.coords = comm_grid.Get_coords(self.rank)

        # Pre-sort all data by time index once at the beginning 
        r_locs, c_locs, t_globs, vals = local_coord_data
        
        if t_globs.size > 0:
            if self.rank == 0:
                print("Pre-sorting local data by time (one-time setup)...")
                
            sort_indices = np.argsort(t_globs, kind='mergesort')
            self.r_locs = r_locs[sort_indices]
            self.c_locs = c_locs[sort_indices]
            self.t_globs = t_globs[sort_indices]
            self.vals = vals[sort_indices]
        else:
            # If this rank has no data, initialize empty arrays
            self.r_locs, self.c_locs, self.t_globs, self.vals = local_coord_data

        # Fast index-based lookup map 
        unique_times, start_indices = np.unique(self.t_globs, return_index=True)
        end_indices = np.append(start_indices[1:], self.t_globs.size)
        
        self.time_slice_map = {t: (start, end) for t, start, end in zip(unique_times, start_indices, end_indices)}

        # MF settings and Hyperparameters  
        self.output_dir = args.output_dir
        self.s = args.periodicity
        self.k = args.n_components
        self.r = args.max_regimes
        self.alpha = args.learning_rate
        self.beta = args.penalty
        self.init_cycles = args.init_cycles
        self.max_iter = args.max_iter
        self.update_freq = args.update_freq
        self.eps = 1e-12
        self.g = 1
        self.last_used_time = np.full(self.r, -1, dtype=np.int64)
        self.cost_history = deque(maxlen=args.max_regimes)
        self.selection_times = []
        self.generation_times = []
        self.ncp_time = 0
        self.initialization_time = 0

    def initialize(self, d_shape: Tuple[int, int], n_total: int):
        t_start_initialization = MPI.Wtime()
        if self.rank == 0: print("Starting 2D Distributed Initialization ")

        self.d, self.n = d_shape, n_total
        d0, d1 = self.d
        dims = self.comm_grid.dims
        P_rows, P_cols = dims[0], dims[1]
        self.u_counts, self.u_displs = get_partition_info(d0, P_rows)
        self.v_counts, self.v_displs = get_partition_info(d1, P_cols)
        self.local_d0 = self.u_counts[self.coords[0]]
        self.local_d1 = self.v_counts[self.coords[1]]

        # Prepare data for NCP from the init window
        init_end = self.s * self.init_cycles
        init_indices = np.where(self.t_globs < init_end)[0]
        init_r_locs = self.r_locs[init_indices]
        init_c_locs = self.c_locs[init_indices]
        init_t_globs = self.t_globs[init_indices]
        init_vals = self.vals[init_indices]
        
        # Create and Distribute an Initial Guess (2D)
        U_init_local = np.random.rand(self.local_d0, self.k).astype(np.float64)
        V_init_local = np.random.rand(self.local_d1, self.k).astype(np.float64)
        W_init = None
        if self.rank == 0:
            init_s = init_end 
            W_init = np.random.rand(init_s, self.k).astype(np.float64)
        W_init = self.comm.bcast(W_init, root=0)
        
        # Run 2D Distributed NCP 
        t_start_ncp = MPI.Wtime()
        self.U_local, self.V_local, W_season = ncp_distributed_2d(
            init_r_locs, init_c_locs, init_t_globs, init_vals,
            U_init_local, V_init_local, W_init,
            self.k, 10,                                   # <--- 10 iterations, can alter
            self.comm_grid, self.comm_row, self.comm_col
        )
        if self.rank == 0:
            self.ncp_time = MPI.Wtime() - t_start_ncp

        # Finalize State 
        self.W = np.zeros((self.r, self.n + self.s, self.k))
        self.R = np.zeros(self.n, dtype=int)
        for r_idx in range(self.r):
            self.W[r_idx, :init_end, :] = W_season
        if self.rank == 0:
            self.initialization_time = MPI.Wtime() - t_start_initialization
            print(f"Distributed initialization finished in {self.initialization_time:.4f}s (NCP took {self.ncp_time:.4f}s).")

    def apply_grad_2d(self, U_local_in, V_local_in, wt, r_locs, c_locs, vals):
        """
        SSMF gradient calculations on 2D grid.
        """
        U_local, V_local = U_local_in.copy(), V_local_in.copy()
        D = np.diag(wt)

        # All ranks must participate in these reductions.
        local_utu = U_local.T @ U_local
        utu = np.empty_like(local_utu)
        self.comm_col.Allreduce(local_utu, utu, op=MPI.SUM)
        local_vtv = V_local.T @ V_local
        vtv = np.empty_like(local_vtv)
        self.comm_row.Allreduce(local_vtv, vtv, op=MPI.SUM)

        # Ranks with no data will have empty arrays and contribute zeros (this shouldn't happen much)
        if r_locs.size > 0:
            grad_U_partial, grad_V_partial = _calculate_partial_gradients_numba(        # partial gradients, parallelized
                U_local, V_local, D, r_locs, c_locs, vals
            )
        else:
            grad_U_partial = np.zeros_like(U_local)
            grad_V_partial = np.zeros_like(V_local)

        # All ranks participate in these gradient reductions.
        grad_U_sparse = np.empty_like(grad_U_partial)
        self.comm_row.Allreduce(grad_U_partial, grad_U_sparse, op=MPI.SUM)              # collect partial gradient info
        grad_V_sparse = np.empty_like(grad_V_partial)
        self.comm_col.Allreduce(grad_V_partial, grad_V_sparse, op=MPI.SUM)

        grad_U_local = grad_U_sparse - (U_local @ D @ vtv @ D)
        grad_V_local = grad_V_sparse - (V_local @ D @ utu @ D)

        # The normalization logic is unchanged.
        U_new, V_new = U_local.copy(), V_local.copy()
        
        grad_u_norm_sq_local = np.sum(grad_U_local**2)
        grad_u_norm = np.sqrt(self.comm_grid.allreduce(grad_u_norm_sq_local, op=MPI.SUM))
        if grad_u_norm > self.eps:
            scale = min(1.0, self.alpha * np.sqrt(self.k) / grad_u_norm)
            U_new += self.alpha * scale * grad_U_local

        grad_v_norm_sq_local = np.sum(grad_V_local**2)
        grad_v_norm = np.sqrt(self.comm_grid.allreduce(grad_v_norm_sq_local, op=MPI.SUM))
        if grad_v_norm > self.eps:
            scale = min(1.0, self.alpha * np.sqrt(self.k) / grad_v_norm)
            V_new += self.alpha * scale * grad_V_local

        U_new = U_new.clip(min=self.eps)
        V_new = V_new.clip(min=self.eps)

        u_norms_sq_local = np.sum(U_new**2, axis=0)
        u_norms_sq = np.empty_like(u_norms_sq_local)
        self.comm_col.Allreduce(u_norms_sq_local, u_norms_sq, op=MPI.SUM)
        u_norms = np.sqrt(u_norms_sq)
        u_norms[u_norms < self.eps] = 1.0

        v_norms_sq_local = np.sum(V_new**2, axis=0)
        v_norms_sq = np.empty_like(v_norms_sq_local)
        self.comm_row.Allreduce(v_norms_sq_local, v_norms_sq, op=MPI.SUM)
        v_norms = np.sqrt(v_norms_sq)
        v_norms[v_norms < self.eps] = 1.0
        
        U_new /= u_norms
        V_new /= v_norms
        wt_new = wt * u_norms * v_norms
        
        return U_new, V_new, wt_new

    def _calculate_cost_vectorized_2d(self, U_local, V_local, W_window, t):
        """
        Data fit cost function.
        """
        window_t_start = t - self.s + 1
        start_slice = np.searchsorted(self.t_globs, window_t_start, side='left')
        end_slice = np.searchsorted(self.t_globs, t, side='right')

        if self.comm_grid.allreduce(end_slice - start_slice, op=MPI.SUM) < 2:
            return 0.0

        window_r_locs = self.r_locs[start_slice:end_slice]
        window_c_locs = self.c_locs[start_slice:end_slice]
        window_vals = self.vals[start_slice:end_slice]
        window_t_globs = self.t_globs[start_slice:end_slice]
        window_t_window = window_t_globs - window_t_start
        has_data = window_r_locs.size > 0
        
        # Vectorized Prediction & Error Calculation (Local) 
        if has_data:
            U_rows = U_local[window_r_locs]
            V_rows = V_local[window_c_locs]
            W_rows = W_window[window_t_window]
            local_v_preds = np.einsum('ij,ij->i', U_rows, W_rows * V_rows)
            local_errors = window_vals - local_v_preds
        else:
            local_errors = np.array([])

        local_sum_err = np.sum(local_errors)
        local_sum_sq_err = np.sum(local_errors**2)
        local_n = float(local_errors.size)

        local_stats = np.array([local_sum_err, local_sum_sq_err, local_n])
        global_stats = np.empty_like(local_stats)
        self.comm_grid.Allreduce(local_stats, global_stats, op=MPI.SUM)
        global_sum_err, global_sum_sq_err, global_n = global_stats
        
        # Prevent division by zero if window has no data points
        if global_n == 0:
            return 0.0
            
        global_mean = global_sum_err / global_n
        global_std = np.sqrt(max(0, (global_sum_sq_err / global_n) - global_mean**2))
        if global_std < self.eps:
            global_std = self.eps

        if has_data:
            local_logprob_sum = np.sum(norm.logpdf(local_errors, loc=global_mean, scale=global_std))
        else:
            local_logprob_sum = 0.0
                
        total_logprob = self.comm_grid.allreduce(local_logprob_sum, op=MPI.SUM)
        final_cost = -1 * total_logprob / np.log(2.0)
        
        return final_cost

    def regime_selection_vectorized_2d(self, t: int):
        """
        Selects the best existing regime by calculating a total cost.
        Total cost = data fit cost + model complexity cost.
        """
        # Array to store the final, total cost for each existing regime.
        costs = np.zeros(self.g)
        
        # 1. Loop through each existing regime to calculate its total cost 
        for ridx in range(self.g):
            
            # Get the temporal factor for this specific regime's window
            W_window = self.W[ridx, t - self.s + 1 : t + 1]

            # 2. Calculate the Data Fit Cost 
            # Call your reusable helper function. This is the probabilistic cost.
            data_cost = self._calculate_cost_vectorized_2d(self.U_local, self.V_local, W_window, t)

            # 3. Calculate the Model Complexity Cost 
            model_cost = utils.compute_model_cost(W_window, 32, self.eps) 

            # 4. The total cost is the sum. 
            costs[ridx] = data_cost + model_cost

        # 5. Find the best regime.
        # np.nanmin is safer if any costs could potentially be NaN
        best_cost = np.nanmin(costs)
        best_ridx = np.nanargmin(costs)
        
        return best_cost, best_ridx

    def regime_generation_2d(self, t, best_current_ridx):
        """
        Generates a candidate regime and calculates cost.
        """
        # Refine factors to create a candidate regime
        Unew_local, Vnew_local = self.U_local.copy(), self.V_local.copy()
        Wnew_season = self.W[best_current_ridx, t - self.s + 1:t + 1].copy()
        
        for _ in range(self.max_iter):
            for i, tt in enumerate(range(t - self.s + 1, t + 1)):
                start, end = self.time_slice_map.get(tt, (0, 0))
                if self.comm_grid.allreduce(end - start) == 0:
                    continue
                
                # Slice the main arrays to get data for this timestep
                r_slice = self.r_locs[start:end]
                c_slice = self.c_locs[start:end]
                v_slice = self.vals[start:end]
                
                Unew_local, Vnew_local, Wnew_season[i] = self.apply_grad_2d(
                    Unew_local, Vnew_local, Wnew_season[i], r_slice, c_slice, v_slice
                )
        
        # Calculate data cost
        data_cost = self._calculate_cost_vectorized_2d(Unew_local, Vnew_local, Wnew_season, t)
        
        # Calculate model cost
        model_cost = utils.compute_model_cost(Wnew_season, 32, self.eps) # Assuming float_cost=32

        # Total cost
        total_cost = data_cost + model_cost

        return total_cost, Unew_local, Vnew_local, Wnew_season

    def calculate_rmse_2d(self, t_forecast, ridx_forecast):
        """
        RMSE calculation on 2D grid.
        """
        if t_forecast >= self.n:
            return None, None, None

        # Get the indices for the forecast time 
        start, end = self.time_slice_map.get(t_forecast, (0, 0))
        num_nz_local = end - start
        
        total_num_nz = self.comm_grid.allreduce(num_nz_local, op=MPI.SUM)
        if total_num_nz == 0:
            return 0.0, 0.0, 0.0

        # Slice the main arrays to get ground truth data 
        truth_r_locs = self.r_locs[start:end]
        truth_c_locs = self.c_locs[start:end]
        truth_vals = self.vals[start:end]

        # Vectorized Prediction and Error Calculation 
        wt_forecast = self.W[ridx_forecast, t_forecast - self.s]
        if not np.any(wt_forecast):
            # Handle cases with uninitialized weights
            return None, None, None

        D_w = np.diag(wt_forecast)
        
        # Predict all values for local non-zero elements at once
        U_rows = self.U_local[truth_r_locs]
        V_rows_T = self.V_local[truth_c_locs] @ D_w.T # Pre-calculate the product
        y_preds = np.einsum('ij,ij->i', U_rows, V_rows_T)

        local_sse_nz = np.sum((y_preds - truth_vals)**2)
        local_sum_sq_y_nz = np.sum(y_preds**2)
        
        # The rest of the distributed RMSE logic
        total_sse_nz = self.comm_grid.allreduce(local_sse_nz, op=MPI.SUM)
        rmse_nonzeros = np.sqrt(max(0, total_sse_nz) / total_num_nz)

        total_sum_sq_y_nz = self.comm_grid.allreduce(local_sum_sq_y_nz, op=MPI.SUM)
        local_utu = self.U_local.T @ self.U_local
        utu = np.empty_like(local_utu)
        self.comm_col.Allreduce(local_utu, utu, op=MPI.SUM)
        local_vtv = self.V_local.T @ self.V_local
        vtv = np.empty_like(local_vtv)
        self.comm_row.Allreduce(local_vtv, vtv, op=MPI.SUM)
        
        total_sum_sq_y = np.trace(D_w @ vtv @ D_w @ utu)
        sse_zeros = max(0, total_sum_sq_y - total_sum_sq_y_nz)

        d0, d1 = self.d
        num_zeros = (d0 * d1) - total_num_nz
        rmse_zeros = np.sqrt(sse_zeros / num_zeros) if num_zeros > 0 else 0.0

        total_sse = total_sse_nz + sse_zeros
        rmse_total = np.sqrt(total_sse / (d0 * d1))

        return rmse_total, rmse_zeros, rmse_nonzeros

    def fit_stream(self, n_total: int, d_shape: Tuple[int, int], rmse_freq: int):
        """
        Fit model on 2D grid of processes.
        """        
        self.initialize(d_shape, n_total)
        self.comm.Barrier()
        rmse_total_list, rmse_zeros_list, rmse_nonzeros_list = [], [], []

        if self.rank == 0: print("Starting 2D Distributed Training Loop...")
        
        cost_history_warmup = self.init_cycles
        init_end = self.s * self.init_cycles
        for t in range(init_end, n_total - 1):

            if self.rank == 0 and t % 100 == 0: print(f"  t = {t}/{n_total}")

            # Propagate W from the previous season
            self.W[:, t] = self.W[:, t - self.s]

            # 1. Select best existing regime 
            # All ranks participate and get the same cost and index.
            t_start_select = MPI.Wtime()
            cost1, ridx1 = self.regime_selection_vectorized_2d(t)
            t_end_select = MPI.Wtime()
            if self.rank == 0:
                self.selection_times.append(t_end_select - t_start_select)
                # Rank 0 updates its cost history
                if np.isfinite(cost1):
                    self.cost_history.append(cost1)

            # 2. Determine if we should consider making new regime. 
            # This decision is made by Rank 0 and broadcast to all other ranks.
            
            should_generate = False # Default decision
            if self.rank == 0:
                # Condition 1: Do we have space for a new regime?
                #has_space = self.g < self.r
                # Condition 2: Is the cost anomalous? (Dynamic Threshold)
                is_poor_fit = False
                if len(self.cost_history) > cost_history_warmup:
                    mean_cost = np.mean(self.cost_history)
                    std_cost = np.std(self.cost_history)
                    # A cost > 2 standard deviations above the mean is a poor fit.
                    dynamic_threshold = mean_cost + 1.5 * std_cost
                    if cost1 > dynamic_threshold:
                        is_poor_fit = True
                        #print(f"t={t}: Poor fit detected (Cost={cost1:.0f} > Threshold={dynamic_threshold:.0f}).")
                
                # Final decision by Rank 0
                if is_poor_fit: 
                    should_generate = True
            
            # Broadcast the decision. All ranks will now have the same value for should_generate.
            should_generate = self.comm.bcast(should_generate, root=0)

            # 3. Generate a candidate new regime (if decision was True) 
            cost2 = np.inf
            Unew_local, Vnew_local, Wnew_season = None, None, None

            if should_generate:
                #if self.rank == 0:
                #    print(f"t={t}: Considering new regime.")
                
                t_start_gen = MPI.Wtime()
                cost2, Unew_local, Vnew_local, Wnew_season = self.regime_generation_2d(t, ridx1)
                t_end_gen = MPI.Wtime()
                if self.rank == 0: self.generation_times.append(t_end_gen - t_start_gen)

            # 4. Decide which regime to use for timestep t 
            if cost2 < cost1 * (1 - self.beta):
                #print(f"t={t}: Considering new regime: cost2 < cost1 * (1 - self.beta)")
                # Accept the new regime 
                if self.g < self.r:
                    # There's a free slot.
                    final_ridx = self.g
                    #if self.rank == 0: print(f"t={t}: *** Creating new regime {final_ridx} (CostNew={cost2:.2f} < CostOld={cost1:.2f}) ***")
                    self.g += 1
                else:
                    # No free slots, evict the least recently used.
                    final_ridx = np.argmin(self.last_used_time)
                    #if self.rank == 0: print(f"t={t}: *** Evicting stale regime {final_ridx}, creating new one (CostNew={cost2:.2f} < CostOld={cost1:.2f}) ***")

                # All ranks update their state to use the new factors.
                self.R[t] = final_ridx
                self.U_local = Unew_local
                self.V_local = Vnew_local
                self.W[final_ridx, t - self.s + 1:t + 1] = Wnew_season
            else:
                #print(f"t={t}: Keep old regime: cost2 >= cost1 * (1 - self.beta)")
                # Keep the old regime 
                final_ridx = ridx1
                self.R[t] = final_ridx
                # The factors self.U_local and self.V_local are NOT changed.
            
            # Update the last used time for the chosen regime.
            self.last_used_time[final_ridx] = t

            # 4. Do gradient update for slice t 
            # This step refines the factors (U, V, and W) using only the data
            # from the current time slice `t`. This is crucial for adapting to
            # the most recent data.
            start, end = self.time_slice_map.get(t, (0,0))
            if self.comm_grid.allreduce(end - start) > 0:
                r_slice = self.r_locs[start:end]
                c_slice = self.c_locs[start:end]
                v_slice = self.vals[start:end]
                
                wt = self.W[final_ridx, t]
                U_updated, V_updated, wt_new = self.apply_grad_2d(
                    self.U_local, self.V_local, wt, r_slice, c_slice, v_slice
                )
                # All ranks update their local factors and the W slice.
                self.U_local = U_updated
                self.V_local = V_updated
                self.W[final_ridx, t] = wt_new

            # 5. Calculate RMSE (at whatever interval you like) 
            if (t + 1) % rmse_freq == 0:
                # The final_ridx from the decision is used for the forecast.
                rmse_t, rmse_z, rmse_nz = self.calculate_rmse_2d(t + 1, final_ridx)
                if self.rank == 0 and rmse_t is not None:
                    rmse_total_list.append(rmse_t)
                    rmse_zeros_list.append(rmse_z)
                    rmse_nonzeros_list.append(rmse_nz)

                    t_str = f"{t:05d}"
                    save_path = Path(self.output_dir) / f"factors_t{t_str}.npz"
                    np.savez_compressed(
                        save_path,
                        t=t,
                        regime=int(self.R[t]),
                        U=self.U_local,
                        V=self.V_local,
                        W=self.W[int(self.R[t])].copy()  # only save the regime used at t
                    )

                    if (t+1) % 100 == 0:
                        print(f"\nRMSE @ t={t+1} (using regime {final_ridx}) ")
                        print(f"Total    : {rmse_t:.6f}\nZeros    : {rmse_z:.6f}\nNon-zeros: {rmse_nz:.6f}\n")
            self.comm.Barrier()

        if self.rank == 0:
            print("Distributed training finished.")
            if rmse_total_list:
                print("\nAverage RMSE Across All Measurements ")
                print(f"Total    : {np.mean(rmse_total_list):.6f}")
                print(f"Zeros    : {np.mean(rmse_zeros_list):.6f}")
                print(f"Non-zeros: {np.mean(rmse_nonzeros_list):.6f}")
                try:
                    output_path = Path(self.output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    np.savetxt(output_path / "rmse_total.txt", rmse_total_list)
                    np.savetxt(output_path / "rmse_zeros.txt", rmse_zeros_list)
                    np.savetxt(output_path / "rmse_nonzeros.txt", rmse_nonzeros_list)
                    print(f"\nRMSE history saved to '{output_path}'")
                except Exception as e:
                    print(f"\nError saving RMSE files: {e}")

            print("\nPerformance Profile ")
            if self.selection_times:
                total_select_time = sum(self.selection_times)
                avg_select_time = total_select_time / len(self.selection_times)
                print(f"Regime Selection : {total_select_time:.4f}s total ({avg_select_time:.6f}s avg over {len(self.selection_times)} calls)")

            if self.generation_times:
                total_gen_time = sum(self.generation_times)
                avg_gen_time = total_gen_time / len(self.generation_times)
                print(f"Regime Generation: {total_gen_time:.4f}s total ({avg_gen_time:.6f}s avg over {len(self.generation_times)} attempts)")

            print(f"NCP initialization took {self.ncp_time:.4f}s")
            print(f"Total initialization took {self.initialization_time:.4f}s")
            print("Regime history: ")
            
            unique, counts = np.unique(self.R, return_counts=True)
            total = counts.sum()
        
            print("\nSummary of self.R (Categorical) ")
            for u, c in zip(unique, counts):
                percent = (c / total) * 100
                print(f"{u}: {c} ({percent:.2f}%)")

            


def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("parquet", help="Parquet file with PU_idx/DO_idx/t_idx/trip_count")
    pa.add_argument("--output_dir", default="out_stream_mpi_2d")
    pa.add_argument("--periodicity", type=int, default=24)
    pa.add_argument("--n_components", type=int, default=10)
    pa.add_argument("--max_regimes", type=int, default=50)
    pa.add_argument("--init_cycles", type=int, default=3)
    pa.add_argument("--learning_rate", type=float, default=0.2)
    pa.add_argument("--penalty", type=float, default=0.05)
    pa.add_argument("--update_freq", type=int, default=1)
    pa.add_argument("--max_iter", type=int, default=1)
    pa.add_argument("--rmse_freq", type=int, default=100)
    pa.add_argument("--grid-rows", type=int, default=None, help="Number of rows in the process grid (manual layout)")
    pa.add_argument("--grid-cols", type=int, default=None, help="Number of columns in the process grid (manual layout)")

    pa.add_argument(
        "--save-forecasts",
        action="store_true", # Boolean flag
        help="Save per-element forecast values for non-zero ground truths. Can be memory intensive."
    )

    args = pa.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dims = [0, 0]

    # Grid dimensions logic
    if args.grid_rows is None and args.grid_cols is None:
        dims = MPI.Compute_dims(size, 2)

    elif args.grid_rows is not None and args.grid_cols is not None:

        if rank == 0 and args.grid_rows * args.grid_cols != size:
            print(f"Layout Error: A {args.grid_rows}x{args.grid_cols} grid requires {args.grid_rows * args.grid_cols} processes, but you provided {size}.")
            comm.Abort(1)

        dims = [args.grid_rows, args.grid_cols]

    elif args.grid_rows is not None:

        if size % args.grid_rows != 0:
            if rank == 0: print(f"Layout Error: Total processes ({size}) is not divisible by specified rows ({args.grid_rows}).")
            comm.Abort(1)

        dims = [args.grid_rows, size // args.grid_rows]

    elif args.grid_cols is not None:
        if size % args.grid_cols != 0:
            if rank == 0: print(f"Layout Error: Total processes ({size}) is not divisible by specified columns ({args.grid_cols}).")
            comm.Abort(1)

        dims = [size // args.grid_cols, args.grid_cols]

    comm_grid = comm.Create_cart(dims, periods=[False, False], reorder=True)
    coords = comm_grid.Get_coords(rank)
    comm_row = comm_grid.Sub([False, True])
    comm_col = comm_grid.Sub([True, False])

    # Metadata Sync 
    global_dims = None
    if rank == 0:
        print(f"Running 2D implementation with a {dims[0]}x{dims[1]} process grid.")
        df_meta = pd.read_parquet(args.parquet, columns=['PU_idx', 'DO_idx', 't_idx'])
        d0, d1, T = (df_meta[c].max() + 1 for c in df_meta.columns)
        global_dims = {'d0': d0, 'd1': d1, 'T': T}
        print(f"Data: d0={d0}, d1={d1}, T={T}")
    global_dims = comm.bcast(global_dims, root=0)
    d0, d1, n_total = global_dims['d0'], global_dims['d1'], global_dims['T']

    # Parallel Data Read 
    u_counts, u_displs = get_partition_info(d0, dims[0])
    v_counts, v_displs = get_partition_info(d1, dims[1])
    my_pu_start, my_pu_end = u_displs[coords[0]], u_displs[coords[0]] + u_counts[coords[0]]
    my_do_start, my_do_end = v_displs[coords[1]], v_displs[coords[1]] + v_counts[coords[1]]

    import pyarrow.parquet as pq
    filters = [
        ('PU_idx', '>=', my_pu_start), ('PU_idx', '<', my_pu_end),
        ('DO_idx', '>=', my_do_start), ('DO_idx', '<', my_do_end)
    ]
    local_df = pq.read_table(args.parquet, filters=filters).to_pandas()

    if not local_df.empty:
        r_locs = (local_df['PU_idx'] - my_pu_start).to_numpy(dtype=np.int64)
        c_locs = (local_df['DO_idx'] - my_do_start).to_numpy(dtype=np.int64)
        t_globs = local_df['t_idx'].to_numpy(dtype=np.int64)
        vals = local_df['trip_count'].to_numpy(dtype=np.float64)
    else:
        r_locs, c_locs, t_globs, vals = (np.array([], dtype=np.int64) for _ in range(4))


    model = SSMF_MPI_2D(comm, comm_grid, comm_row, comm_col, (r_locs, c_locs, t_globs, vals), args)
    try:
        model.fit_stream(n_total, (d0, d1), args.rmse_freq)    
    finally:
        comm.Barrier()

    if rank == 0: print("Execution finished successfully.")

if __name__ == "__main__":
    main()
