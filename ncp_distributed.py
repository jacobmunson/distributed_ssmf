import numpy as np
from mpi4py import MPI
import numba

@numba.njit(cache=True)
def mttkrp_u(r_locs, c_globs, t_globs, vals, V, W_season, U_local_shape):
    """Computes the MTTKRP for the U factor at compiled speed."""
    MB_local = np.zeros(U_local_shape, dtype=np.float64)
    for i in range(len(vals)):
        r, c, t, v = r_locs[i], c_globs[i], t_globs[i], vals[i]
        MB_local[r, :] += v * (V[c, :] * W_season[t, :])
    return MB_local

@numba.njit(cache=True)
def mttkrp_v(r_locs, c_globs, t_globs, vals, U_local, W_season, V_shape):
    """Computes the MTTKRP for the V factor at compiled speed."""
    MB_local = np.zeros(V_shape, dtype=np.float64)
    for i in range(len(vals)):
        r, c, t, v = r_locs[i], c_globs[i], t_globs[i], vals[i]
        MB_local[c, :] += v * (U_local[r, :] * W_season[t, :])
    return MB_local

@numba.njit(cache=True)
def mttkrp_w(r_locs, c_globs, t_globs, vals, U_local, V, W_shape):
    """Computes the MTTKRP for the W factor at compiled speed."""
    MB_local = np.zeros(W_shape, dtype=np.float64)
    for i in range(len(vals)):
        r, c, t, v = r_locs[i], c_globs[i], t_globs[i], vals[i]
        MB_local[t, :] += v * (U_local[r, :] * V[c, :])
    return MB_local

def ncp_distributed(
    r_locs, c_globs, t_globs, vals, 
    d_shape: tuple,
    s: int, k: int, maxit: int,
    comm_grid: MPI.Comm,
    comm_row: MPI.Comm,
    comm_col: MPI.Comm,
    u_counts: list,
    v_counts: list,
    coords: tuple,
    U_init_local: np.ndarray, 
    V_init: np.ndarray,       
    W_init: np.ndarray,       
    eps: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a distributed Non-negative CP decomposition.

    Returns:
        A tuple containing (U_local, V_global, W_global_season).
    """
    # 1. Setup and Initialization 
    V = V_init.copy()
    W_season = W_init.copy()
    U_local = U_init_local.copy()    
    
    factors = [U_local, V, W_season]

    # Initialize the list of Gram matrices ('squared' factors).
    # Asq[0] is for U, which is partitioned, so we'll compute its global
    # Gram matrix inside the loop. We initialize it to None.
    # Asq[1] and Asq[2] can be computed locally since V and W are replicated.
    Asq = [
        None,             # Placeholder for global U.T @ U
        V.T @ V,          # Gram matrix for V
        W_season.T @ W_season # Gram matrix for W_season
    ] 
    
    # 2. Main Iterative Loop 
    for iteration in range(maxit):
        # 2a. Update U (Partitioned Factor) 
        # 2a. Update U (Partitioned Factor, A[0]) 
        
        # Get the current global factors and their Gram matrices
        # U_local is factors[0], V is factors[1], W_season is factors[2]
        U_local, V, W_season = factors[0], factors[1], factors[2]
        V_sq, W_sq = Asq[1], Asq[2]
        
        # (i) Compute Bsq = W_sq * V_sq.
        # This is a small (k x k) matrix product, identical on all ranks.
        Bsq = W_sq * V_sq  # Element-wise product
        
        # (ii) Compute the local Matricized Tensor Times Khatri-Rao Product (MTTKRP).
        # This is the core memory-saving step. We iterate through our local sparse
        # triplets instead of forming a dense tensor.
        #MB_local = np.zeros_like(U_local)
        MB_local = mttkrp_u(r_locs, c_globs, t_globs, vals, V, W_season, U_local.shape)

        
        # (iii) Perform the local gradient update.
        # All computations up to this point have been local, requiring NO communication.
        L = np.linalg.norm(Bsq) + 1e-9 
        Gn_local = U_local @ Bsq - MB_local
        U_local = np.maximum(0, U_local - Gn_local / L)
        U_local = U_local.clip(min=eps) 
        
        # (iv) Synchronize the Gram matrix for the next iteration.
        # Each rank computes its local contribution to the Gram matrix.
        Asq_local_0 = U_local.T @ U_local
        
        # Create a buffer to receive the global result.
        Asq_global_0 = np.empty_like(Asq_local_0)
        
        # Sum the contributions from all ranks to get the final global Gram matrix.
        # This is the ONLY communication step needed for the U update.
        comm_grid.Allreduce(Asq_local_0, Asq_global_0, op=MPI.SUM)
        
        # (v) Store the updated factors for the next step in the loop.
        factors[0] = U_local
        Asq[0] = Asq_global_0

        # 2b. Update V (Replicated Factor, A[1]) 
        
        # Get the current global factors and their Gram matrices
        # Note: Asq[0] is now the global version from the previous step
        U_local, V, W_season = factors[0], factors[1], factors[2]
        U_sq_global, W_sq = Asq[0], Asq[2]
        Bsq = W_sq * U_sq_global
        
        MB_local = mttkrp_v(r_locs, c_globs, t_globs, vals, U_local, W_season, V.shape)
        MB_global = np.empty_like(MB_local)
        comm_grid.Allreduce(MB_local, MB_global, op=MPI.SUM)
        
        # (iv) Perform the global gradient update.
        # Now that every rank has the same global MB, this update is
        # performed identically on all ranks.
        L = np.linalg.norm(Bsq) + 1e-9 
        Gn = V @ Bsq - MB_global
        V = np.maximum(0, V - Gn / L)
        V = V.clip(min=eps)
        
        # (v) Store the updated factor and its new Gram matrix.
        # This is a local operation; no communication needed here.
        factors[1] = V
        Asq[1] = V.T @ V    

        # 2c. Update W (Replicated Factor, A[2]) 
        
        # Get the current factors and Gram matrices.
        # Asq[0] and Asq[1] are now the global versions from previous steps.
        U_local, V, W_season = factors[0], factors[1], factors[2]
        U_sq_global, V_sq = Asq[0], Asq[1]
        
        # (i) Compute Bsq = V_sq * U_sq_global.
        # This is identical on all ranks.
        Bsq = V_sq * U_sq_global
        
        # (ii) Compute the LOCAL contribution to the MTTKRP for W.
        MB_local = mttkrp_w(r_locs, c_globs, t_globs, vals, U_local, V, W_season.shape)

        # (iii) Aggregate the partial results into the GLOBAL MTTKRP.
        MB_global = np.empty_like(MB_local)
        comm_grid.Allreduce(MB_local, MB_global, op=MPI.SUM)
        
        # (iv) Perform the identical global gradient update on all ranks.
        L = np.linalg.norm(Bsq) + 1e-9 
        Gn = W_season @ Bsq - MB_global
        W_season = np.maximum(0, W_season - Gn / L)
        W_season = W_season.clip(min=eps)
        
        # (v) Store the updated factor and its new Gram matrix locally.
        factors[2] = W_season
        Asq[2] = W_season.T @ W_season
        
    # 3. Return Final Factors 
    # The final U_local is already on the correct rank.
    # The final V and W_season are identical on all ranks.
    U_local = factors[0]
    V_global = factors[1]
    W_global_season = factors[2]

    return U_local, V_global, W_global_season
