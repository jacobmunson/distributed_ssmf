"""
Shifting Seasonal Matrix Factorization on a parquet tuple stream.
Keeps features from the original dense version. 
Processes data as input tuples vs. dense (explicit 0 representations) tensor.

To run:
python ssmf_tuples.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet
"""
from __future__ import annotations
import argparse, time, warnings
from collections import deque, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Deque, Iterator, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from tqdm import tqdm
import numba
from joblib import Parallel, delayed
import ncp, utils                          

USE_NUMBA = True

# Helper functions
class TupleWindow:
    """
    Buffer to store the last 's' time slices as lists of tuples.
    """

    def __init__(self, d_shape: Tuple[int, int], s: int):
        self.d_shape = d_shape
        self.s = s
        self.buf: Deque[List[Tuple[int, int, float]]] = deque(maxlen=s)

    def push(self, triples: List[Tuple[int, int, float]]):
        self.buf.append(triples)

    def full(self) -> bool:
        return len(self.buf) == self.s

    def dense(self) -> np.ndarray:
        """Dense (d0, d1, s) tensor view of the current window."""
        d0, d1 = self.d_shape
        X = np.zeros((d0, d1, self.s), dtype=float)
        for t_local, triples in enumerate(self.buf):
            if triples:
                r, c, v = zip(*triples)
                X[r, c, t_local] = v
        return X

def rmse_components_tuple(forecast: np.ndarray,
                          triples: List[Tuple[int, int, float]],
                          d_shape: Tuple[int, int]):
    """
    Compare dense forecast (d0×d1) with sparse ground-truth tuples.
    Returns (rmse_total, rmse_zeros, rmse_nonzeros).    <- more nuanced view of RMSE
    """
    d0, d1 = d_shape
    total_elements = d0 * d1
    
    # 1. Calculate error on non-zero elements
    sse_nonzero = 0.0
    num_nonzero = len(triples)
    if num_nonzero > 0:
        r, c, v = zip(*triples)
        r, c = np.array(r), np.array(c) # Convert to numpy arrays for indexing
        v = np.array(v)
        
        diff_nonzero = forecast[r, c] - v
        sse_nonzero = np.sum(diff_nonzero ** 2)
        rmse_nonzeros = np.sqrt(sse_nonzero / num_nonzero)
    else:
        rmse_nonzeros = 0.0
        
    # 2. Calculate error on zero elements
    num_zeros = total_elements - num_nonzero
    sse_zeros = 0.0
    if num_zeros > 0:
        # The error on a zero element is (0 - forecast_value)^2 = forecast_value^2
        # Sum of all squared forecast values
        sse_forecast_total = np.sum(forecast ** 2)
        
        # Sum of squared forecast values ONLY at the non-zero locations
        sse_forecast_at_nonzero = np.sum(forecast[r, c] ** 2) if num_nonzero > 0 else 0
        
        # By subtraction, we get the sum of squared forecast values at zero locations
        sse_zeros = sse_forecast_total - sse_forecast_at_nonzero
        rmse_zeros = np.sqrt(sse_zeros / num_zeros)
    else:
        rmse_zeros = 0.0
        
    # 3. Calculate total RMSE
    total_sse = sse_nonzero + sse_zeros
    rmse_total = np.sqrt(total_sse / total_elements)
    
    return rmse_total, rmse_zeros, rmse_nonzeros

def parquet_slice_stream(path: str | Path,
                         *,
                         cols=("PU_idx", "DO_idx", "t_idx", "trip_count")
                         ) -> Iterator[Tuple[int, List[Tuple[int, int, float]]]]:
    """Get (t_idx, [(row,col,val), …]) in ascending t_idx order."""
    df = (
        pd.read_parquet(path, columns=cols)
          .astype({"PU_idx": "int32", "DO_idx": "int32",
                   "t_idx": "int32", "trip_count": "float32"})
          .sort_values("t_idx")
    )
    for t_idx, grp in df.groupby("t_idx", sort=False):
        yield int(t_idx), list(zip(grp.PU_idx, grp.DO_idx, grp.trip_count))



class SSMF:
    """
    Shifting Seasonal Matrix Factorization for sparse tuple streams.
    Includes full regime creation and online update logic.
    """
    def __init__(self, triplet_dict, periodicity, n_components,
                 max_regimes=100, epsilon=1e-12,
                 alpha=0.1, beta=0.05, max_iter=5, update_freq=1,
                 init_cycles=3, float_cost=32):

        assert periodicity  > 0 and n_components > 1
        assert max_regimes  > 0 and init_cycles  > 1

        self.triplet_dict = triplet_dict
        self.s = periodicity
        self.k = n_components
        self.r = max_regimes
        self.g = 1  # of current regimes

        self.eps = epsilon
        self.alpha = alpha
        self.beta = beta
        self.init_cycles = init_cycles
        self.max_iter = max_iter
        self.update_freq = update_freq
        self.float_cost = float_cost
        # These will be set by initialize()
        self.d = (0,0)
        self.n = 0
        self.U = []
        self.W = np.array([])
        self.R = np.array([])
        self.output_dir = "out"

    def initialize(self, X_window: np.ndarray, n_total: int):
        """
        Initializes factors from a dense tensor built from the first few seasons.
        """
        self.d = X_window.shape[:-1]
        self.n = n_total

        self.U = [np.zeros((i, self.k)) for i in self.d]
        self.W = np.zeros((self.r, self.s + self.n, self.k))
        self.R = np.zeros(self.n, dtype=int)

        factor = ncp.ncp(X_window, self.k, maxit=3, verbose=False)      # <- NCP call
        self.W[0, :self.s] = factor[-1] # Only initialize for regime 0

        # Normalization
        for i in range(len(self.d)):
            weights = np.sqrt(np.sum(factor[i] ** 2, axis=0))
            weights[weights == 0] = self.eps # Avoid division by zero
            self.U[i] = factor[i] @ np.diag(1 / weights)
            self.W[0, :self.s] = self.W[0, :self.s] @ np.diag(weights)

    def regime_generation_tuples(self, window_triples: Sequence[List[Tuple]], t: int, ridx: int, max_iter: int):
        """
        Creates a candidate new regime by fitting new factors to the current data window.
        This is the sparse-data equivalent of the original 'regime_generation'.
        """
        # 1. Initialize new factors by copying current
        #    (U_new is a list [U0, U1], W_new is a (s, k) numpy array)
        U_new = deepcopy(self.U)
        W_new = self.W[ridx, t - self.s + 1 : t + 1].copy()

        # 2. Refine candidate factors over entire window
        # Pre-compute Gram matrices once for this pass       
        if USE_NUMBA:
            for _ in range(max_iter):
                for i, triples_in_slice in enumerate(window_triples):
                    if triples_in_slice:
                        r, c, v = zip(*triples_in_slice)
                        U_new[0], U_new[1], W_new[i] = self.apply_grad_numba(
                            U_new[0], U_new[1], W_new[i],
                            np.array(r, dtype=np.int32), np.array(c, dtype=np.int32), np.array(v),
                            alpha=0.5, eps=self.eps)
        else: 
            gram1 = U_new[1].T @ U_new[1]
            gram0 = U_new[0].T @ U_new[0]
            for i, triples_in_slice in enumerate(window_triples):
                U_new[0], U_new[1], W_new[i] = self.apply_grad_sparse(
                    U_new, W_new[i], triples_in_slice, self.d, alpha=0.5, eps=self.eps, gram_matrices=(gram1, gram0) 
                )

        # 3. Calculate the total cost of this newly refined regime
        # a) Coding cost (using the probabilistic method)
        coding_cost = 0.0
        for i, triples_in_slice in enumerate(window_triples):
            u, v, w = U_new[0], U_new[1], W_new[i]
            coding_cost += utils.coding_cost_tuples_probabilistic(triples_in_slice, u, v, w, self.d)

        # b) Model cost (the cost of storing the new W factor)
        model_cost = utils.compute_model_cost(W_new, self.float_cost, self.eps)
        total_cost = coding_cost + model_cost

        return total_cost, U_new, W_new

    def fit_stream(self, n_total: int, d_shape: Tuple[int, int]):
        """
        Main online loop, including regime selection/generation logic.
        """
        s = self.s
        forecasts, rmse_total_list, rmse_zeros_list, rmse_nonzeros_list = [], [], [], []

        # 1. Initialization 
        print("Initializing model from first cycles...")
        init_end_t = s * self.init_cycles
        init_window_data = self.make_window(init_end_t - 1, init_end_t) # A single large window
        
        init_tensor = np.zeros((*d_shape, init_end_t))                  # Mirror original SSMF initialization logic
        for i, (_, triples) in enumerate(init_window_data):
            if triples:
                r, c, v = zip(*triples)
                init_tensor[r, c, i] = v
        
        X_fold = [init_tensor[..., i*s:(i+1)*s] for i in range(self.init_cycles)]
        X_fold_avg = np.array(X_fold).sum(axis=0) / self.init_cycles
        self.initialize(X_fold_avg, n_total)

        # 2. Online Loop
        # for t in tqdm(range(s, n_total - 1), unit="slice", desc="Streaming fit"):  # if you want a progress bar
        for t in range(s, n_total - 1): 
            self.W[:, t] = self.W[:, t - s] # Seasonal weights

            window_data = self.make_window(t, s)
            window_triples = [triples for _, triples in window_data]
            
            # a) Cost of using an existing regime
            cost1, ridx1 = self.regime_selection_vectorized(window_triples, t)

            # b) Cost of generating a new one (optional)
            cost2 = np.inf
            if t % self.update_freq == 0:
                cost2, Unew, Wnew = self.regime_generation_tuples(window_triples, t, ridx1, self.max_iter)

            # c) Create new regime or use existing?
            if cost1 + self.beta * cost1 < cost2:           # Use existing regime
                self.R[t] = ridx1
            else:                                           # Create a new regime
                if self.g < self.r:
                    #print(f"\nTime {t}: New regime {self.g} created.")
                    self.R[t] = self.g
                    self.U = Unew
                    self.W[self.g, t-s+1 : t+1] = Wnew
                    self.g += 1
                else:
                    # Max regimes reached, fall back to best existing
                    self.R[t] = ridx1
                    #if not self.g == 1:
                        #warnings.warn(f"Time {t}: # of regimes exceeded the limit ({self.r})")
            
            # d) Final gradient update on the current time slice using the chosen regime
            if USE_NUMBA:
                final_ridx = self.R[t]
                current_wt = self.W[final_ridx, t]
                current_triples = window_triples[-1]

                if current_triples: # Only update if there's data (there should be)
                    r, c, v = zip(*current_triples)
                    self.U[0], self.U[1], self.W[final_ridx, t] = self.apply_grad_numba(
                        self.U[0], self.U[1], current_wt,
                        np.array(r, dtype=np.int32), np.array(c, dtype=np.int32), np.array(v),
                        alpha=self.alpha, eps=self.eps
                    )
            else:
                final_ridx = self.R[t]
                current_wt = self.W[final_ridx, t]
                current_triples = window_triples[-1]
                self.U[0], self.U[1], self.W[final_ridx, t] = self.apply_grad_sparse(       # <- do the gradient update
                    self.U, current_wt, current_triples, d_shape, self.alpha, self.eps
                )

            # 3. Forecast and Evaluate RMSE 
            forecast_t = self.forecast(self.R[t], t, steps_ahead=1)
            forecasts.append(forecast_t)

            next_triples = self.triplet_dict.get(t + 1, [])
            total, zeros, nonzeros = rmse_components_tuple(forecast_t, next_triples, d_shape)
            rmse_total_list.append(total)
            rmse_zeros_list.append(zeros)
            rmse_nonzeros_list.append(nonzeros)

        #------- 4. Save results------------------------------------
        self.save_results(forecasts, rmse_total_list, rmse_zeros_list, rmse_nonzeros_list)

    def regime_selection_vectorized(self, window_triples: Sequence[List[Tuple]], t: int):
        """
        Vectorization to make per regime calculation fast.
        joblib to run the work for multiple regimes in parallel.
        """
        # 1. Aggregate all non-zero data from the entire window.
        # This setup is done once before we evaluate any regimes.
        all_r, all_c, all_v, all_t_local = [], [], [], []
        for i, triples in enumerate(window_triples):
            if not triples: continue
            r, c, v = zip(*triples)
            all_r.extend(r); all_c.extend(c); all_v.extend(v)
            all_t_local.extend([i] * len(r))

        if not all_r: return 0.0, 0
        all_r, all_c, all_v, all_t_local = map(np.array, [all_r, all_c, all_v, all_t_local])
        U0, U1 = self.U

        # 2. Work for a single regime.
        def _cost_for_one_regime(ridx):
            W_window = self.W[ridx, t - self.s + 1 : t + 1]
            
            # Vectorized prediction and cost calculation
            U0_rows = U0[all_r]
            U1_rows = U1[all_c]
            W_rows = W_window[all_t_local]
            v_preds = np.einsum('ij,ij->i', U0_rows, W_rows * U1_rows)
            errors = all_v - v_preds
            
            if errors.size < 2: return 0.0
            
            error_mean = errors.mean()
            error_std = errors.std()
            if error_std < 1e-9: error_std = 1e-9
                
            logprob = norm.logpdf(errors, loc=error_mean, scale=error_std)
            return -1 * logprob.sum() / np.log(2.0)

        # 3. Use parallel resources?
        if self.g <= 2: 
            all_costs = [_cost_for_one_regime(r) for r in range(self.g)]
        else:                                           
            # When g > 2, use joblib to run in parallel.
            all_costs = Parallel(n_jobs=-1, backend="threading")(
                delayed(_cost_for_one_regime)(r) for r in range(self.g)
            )

        E = np.array(all_costs)
        best_idx = np.argmin(E)
        return E[best_idx], best_idx

    @staticmethod
    def apply_grad_sparse(U_in, wt, triples, d_shape, alpha, eps, gram_matrices=None):
        U0, U1 = U_in[0].copy(), U_in[1].copy()
        U = [U0, U1]
        k = U0.shape[1]
        D = np.diag(wt)

        if triples:
            r, c, v = zip(*triples)
            Xt_csr = sparse.coo_matrix((v, (r, c)), shape=d_shape).tocsr()
            grad0 = Xt_csr.dot(U1).dot(D)
            grad1 = Xt_csr.T.dot(U0).dot(D)
        else:
            grad0, grad1 = np.zeros_like(U0), np.zeros_like(U1)

        grad0 -= U0 @ D @ (U1.T @ U1) @ D
        grad1 -= U1 @ D @ (U0.T @ U0) @ D
        # If Gram matrices are not provided, compute. Otherwise, use provided.
        if gram_matrices is None:
            gram1 = U1.T @ U1
            gram0 = U0.T @ U0
        else:
            gram1, gram0 = gram_matrices

        grad0 -= U0 @ D @ gram1 @ D
        grad1 -= U1 @ D @ gram0 @ D

        wt_new = wt.copy()
        for i, grad_i in enumerate([grad0, grad1]):
            gnorm = np.linalg.norm(grad_i)
            if gnorm > eps:
                grad_i *= min(1, alpha * np.sqrt(k) / gnorm)
                U[i] += alpha * grad_i
                
                weights = np.linalg.norm(U[i], axis=0)
                weights[weights == 0] = eps
                U[i] = U[i] @ np.diag(1 / weights)
                U[i].clip(min=eps, out=U[i])
                wt_new *= weights
        return U[0], U[1], wt_new

    @staticmethod
    @numba.njit(cache=True)
    def apply_grad_numba(U0, U1, wt, r_idx, c_idx, v_vals, alpha, eps):
        """
        A Numba-jitted version with the update.
        """
        k = U0.shape[1]
        
        # Gradient Calculation 
        grad0 = np.zeros_like(U0)
        grad1 = np.zeros_like(U1)
        D = np.diag(wt)
        
        # Parallelized with Numba
        for i in numba.prange(len(v_vals)):
            r, c, v = r_idx[i], c_idx[i], v_vals[i]
            grad0[r, :] += v * (U1[c, :] @ D)
            grad1[c, :] += v * (U0[r, :] @ D)
            
        gram1 = U1.T @ U1
        gram0 = U0.T @ U0
        grad0 -= U0 @ D @ gram1 @ D
        grad1 -= U1 @ D @ gram0 @ D

        # Create copies to modify
        U0_new, U1_new = U0.copy(), U1.copy()

        # Update and Normalize U0 
        gnorm0 = np.linalg.norm(grad0)
        if gnorm0 > eps:
            grad0 *= min(1, alpha * np.sqrt(k) / gnorm0)
            U0_new += alpha * grad0
        
        u_weights = np.sqrt(np.sum(U0_new**2, axis=0))
        u_weights[u_weights < eps] = 1.0
        U0_new /= u_weights
        U0_new = np.maximum(U0_new, eps)

        # Update and Normalize U1 
        gnorm1 = np.linalg.norm(grad1)
        if gnorm1 > eps:
            grad1 *= min(1, alpha * np.sqrt(k) / gnorm1)
            U1_new += alpha * grad1

        v_weights = np.sqrt(np.sum(U1_new**2, axis=0))
        v_weights[v_weights < eps] = 1.0
        U1_new /= v_weights
        U1_new = np.maximum(U1_new, eps)

        # Update W
        wt_new = wt * u_weights * v_weights
        
        return U0_new, U1_new, wt_new

    def forecast(self, ridx, current_time, steps_ahead=1):
        U, V = self.U
        future_t = current_time + steps_ahead
        wt = self.W[ridx, future_t - self.s]
        return U @ np.diag(wt) @ V.T

    def make_window(self, t: int, s: int) -> List[Tuple[int, List[Tuple[int,int,float]]]]:
        return [(tt, self.triplet_dict.get(tt, [])) for tt in range(t - s + 1, t + 1)]

    def save_results(self, forecasts, rmse_total, rmse_zeros, rmse_nonzeros):
        print("\n--- Final Results")
        print("Average RMSE (total):", np.mean(rmse_total))
        print("Average RMSE (zeros):", np.mean(rmse_zeros))
        print("Average RMSE (nonzeros):", np.mean(rmse_nonzeros))
        
        out = Path(self.output_dir)
        np.save(out / "ssmf_forecasts.npy", np.array(forecasts))
        np.savetxt(out / "rmse_total_ssmf.txt",   np.array(rmse_total))
        np.savetxt(out / "rmse_zeros_ssmf.txt",  np.array(rmse_zeros))
        np.savetxt(out / "rmse_nonzeros_ssmf.txt",np.array(rmse_nonzeros))

        np.save(out / 'U.npy', self.U[0])
        np.save(out / 'V.npy', self.U[1])
        np.save(out / 'W.npy', self.W)
        np.savetxt(out / 'R.txt', self.R, fmt='%d')
        print(f"All results saved to {self.output_dir}")



# Main()
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("parquet", help="Parquet file with PU_idx/DO_idx/t_idx/trip_count")
    pa.add_argument("--output_dir", default="out_stream")
    pa.add_argument("--periodicity", type=int, default=24)
    pa.add_argument("--n_components", type=int, default=10)
    pa.add_argument("--max_regimes", type=int, default=50)
    pa.add_argument("--max_iter", type=int, default=1)
    pa.add_argument("--learning_rate", type=float, default=0.2)
    pa.add_argument("--penalty", type=float, default=0.05)
    pa.add_argument("--float_cost", type=int, default=32)
    pa.add_argument("--update_freq", type=int, default=1)
    pa.add_argument("--init_cycles", type=int, default=3)
    args = pa.parse_args()

    utils.make_directory(args.output_dir)

    df = pd.read_parquet(args.parquet,
                        columns=["PU_idx", "DO_idx", "t_idx", "trip_count"])\
            .astype({"PU_idx":"int32","DO_idx":"int32",
                    "t_idx":"int32","trip_count":"float32"})
    d1 = df.PU_idx.max() + 1
    d2 = df.DO_idx.max() + 1
    T  = df.t_idx.max() + 1
    print(f"Data: d1={d1}, d2={d2}, T={T}")
    
    triplet_dict = defaultdict(list)
    for r in df.itertuples(index=False):
        triplet_dict[r.t_idx].append((r.PU_idx, r.DO_idx, r.trip_count))

    # Build model
    model = SSMF(
        triplet_dict=triplet_dict,
        periodicity=args.periodicity,
        n_components=args.n_components,
        max_regimes=args.max_regimes,
        max_iter=args.max_iter,
        alpha=args.learning_rate,
        beta=args.penalty,
        update_freq=args.update_freq,
        float_cost=args.float_cost,
        init_cycles=args.init_cycles,
    )
    model.output_dir = args.output_dir

    # Streaming fit
    model.fit_stream(T, (d1, d2))              
    print("Saved results to", args.output_dir)

if __name__ == "__main__":
    main()