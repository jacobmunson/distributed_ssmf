""" 
SSMF: Shifting Seasonal Matrix Factorization 

Slight alteration of https://github.com/kokikwbt/ssmf/tree/main designed to accumulate forecasts at each step.

To run: python ssmf_forecast.py --periodicity 24 --max_regimes 50 --dataset taxi_yellow_green_rideshare_distinct_march_to_apr2020_tensor.npy
"""
import argparse
import warnings
from copy import deepcopy
import time
import numpy as np
from tqdm import trange

try:
    import ncp
    import utils
except:
    from . import ncp
    from . import utils

class SSMF:
    def __init__(self, periodicity, n_components,
                 max_regimes=100, epsilon=1e-12,
                 alpha=0.1, beta=0.05, max_iter=5, update_freq=1,
                 init_cycles=3, float_cost=32):

        assert periodicity  > 0
        assert n_components > 1
        assert max_regimes  > 0
        assert init_cycles  > 1

        self.s = periodicity
        self.k = n_components
        self.r = max_regimes
        self.g = 1  # of regimes

        self.eps = epsilon  # zero threshold
        self.alpha = alpha  # learning rate
        self.beta = beta  # A larger value may create more regimes
        self.init_cycles = init_cycles
        self.max_iter = max_iter
        self.update_freq = update_freq
        self.float_cost = float_cost

    def initialize(self, X):

        self.d = X.shape[:-1]
        self.n = X.shape[-1]
        #print("initialize with (self.d,self.n,self.k) = (" + str(self.d) + "," + str(self.n) + "," + str(self.k) + ")")
        
        # U(t) and V(t)
        self.U = [np.zeros((i, self.k)) for i in self.d]
        #for i in self.d:
        #    print("i = ", i, " d = ", self.d)

        # Full history of W(t)
        self.W = np.zeros((self.r, self.s + self.n, self.k))

        # Regime history
        self.R = np.zeros(self.n, dtype=int)

        # Operation history
        self.O = np.zeros(self.n, dtype=int)
        
        # Estimate the initial factors
        X_fold = [X[..., i*self.s:(i+1)*self.s] for i in range(self.init_cycles)]
        X_fold = np.array(X_fold).sum(axis=0) / self.init_cycles

        import time, sys

        t0 = time.time()

        try:
            factor = ncp.ncp(X_fold, self.k, maxit=3, verbose=True)
            print("ncp → returned in %.2fs" % (time.time() - t0), flush=True)
        except SystemExit as e:
            print("‼ ncp.ncp called sys.exit(", e.code, ")", file=sys.stderr, flush=True)
            # now re-raise (or replace factor with a dummy) so the rest of your code can run:
            raise

        self.W[:, :self.s] = factor[-1]

        # Normalization
        for i in range(len(self.d)):
            weights = np.sqrt(np.sum(factor[i] ** 2, axis=0))            
            self.U[i] = factor[i] @ np.diag(1 / weights)
            self.W[:, :self.s] = self.W[:, :self.s] @ np.diag(weights)

    @staticmethod
    def apply_grad(U, wt, Xt, alpha, eps):

        U0, U1 = U        # initialize 
        D = np.diag(wt)
        k = U0.shape[1]

        grad = [
            Xt @ U1 @ D - U0 @ D @ (U1.T @ U1) @ D,
            Xt.T @ U0 @ D - U1 @ D @ (U0.T @ U0) @ D
        ]

        wt_new = np.copy(wt)

        for i in range(2):

            # Smooth update
            grad[i] *= min(1, alpha * np.sqrt(k) / np.sqrt(np.sum(grad[i] ** 2)))
            U[i] += alpha * grad[i]

            # Normalization
            weights = np.sqrt(np.sum(U[i] ** 2, axis=0))
            U[i] = U[i] @ np.diag((1 / weights))
            U[i] = U[i].clip(min=eps, max=None)
            wt_new = wt_new * weights

        return U[0], U[1], wt_new

    @staticmethod
    def reconstruct(U, V, W):
        Y = np.zeros((U.shape[0], V.shape[0], W.shape[0]))
        for t, wt in enumerate(W):
            Y[..., t] = U @ np.diag(wt) @ V.T

        return Y

    def fit(self, X):

        print("fit")

        n = X.shape[-1]
        print("fit: n = " + str(n))
        elapsed_time = np.zeros(n)

        for t in range(self.s, n):
            print('\nt=', t)

            tic = time.process_time()

            Xc = X[..., t-self.s:t]
            self.update(Xc, t)  # Algorithm 1

            toc = time.process_time()
            elapsed_time[t] = toc - tic

        return elapsed_time

    def update(self, X, t, verbose=0):      # gets called inside test()
        """ Algorithm 1 in the paper

            X: current tensor (u, v, s)
            t: current time point
        """
        cost1 = cost2 = np.inf
        self.W[:, t] = self.W[:, t - self.s]  # Copy

        cost1, ridx1 = self.regime_selection(X, t)

        if t % self.update_freq == 0:
            cost2, Unew, Wnew = self.regime_generation(X, t, ridx1, self.max_iter)

        if verbose > 0:
            print('RegimeSelection', cost1 + self.beta * cost1, ridx1)
            print('RegimeGeneration', cost2, self.g,
                'diff=', cost2 - (cost1 + self.beta * cost1))

        if cost1 + self.beta * cost1 < cost2:
            self.R[t] = ridx1

        else:
            if self.g < self.r:
                self.R[t] = self.g
                self.U = Unew
                self.W[self.g, t - self.s + 1: t + 1] = Wnew
                self.g += 1
            else:
                self.R[t] = ridx1
                if not self.g == 1:
                    warnings.warn("# of regimes exceeded the limit")

        wt = self.W[self.R[t], t]
        Xt = X[..., -1]

        self.U[0], self.U[1], self.W[self.R[t], t] = self.apply_grad(
            self.U, wt, Xt, self.alpha, self.eps)

        # Non-negative constraint
        assert self.U[0].min() >= 0
        assert self.U[1].min() >= 0
        assert self.W.min() >= 0

    def regime_selection(self, X, t):

        U, V = self.U
        n = X.shape[-1]
        Y = np.zeros(X.shape)
        E = np.zeros(self.g)

        for i in range(self.g):
            Wi = self.W[i, t - n + 1:t + 1]
            Y = self.reconstruct(U, V, Wi)
            E[i] = utils.compute_coding_cost(X, Y, self.float_cost)

        best_regime_index = np.argmin(E)
        best_coding_cost  = E[best_regime_index]

        return best_coding_cost, best_regime_index

    def regime_generation(self, X, t, ridx, max_iter=1):

        # Initialize a new W with the nearest components
        n = X.shape[-1]
        U = deepcopy(self.U[0])
        V = deepcopy(self.U[1])
        W = np.zeros((self.s, self.k))
        W = self.W[ridx, t - self.s + 1:t + 1]
        
        # Fitting
        for _ in range(max_iter):
            for tt in range(n):
                U, V, W[tt] = self.apply_grad([U, V], W[tt], X[..., tt], 0.5, self.eps)


        Y = self.reconstruct(U, V, W)
        E = utils.compute_coding_cost(X, Y, self.float_cost)
        E += utils.compute_model_cost(W, self.float_cost, self.eps)

        return E, [U, V], W

    def forecast(self, ridx, current_time, steps_ahead=1):
        U, V = self.U
        future = current_time + steps_ahead
        wt     = self.W[ridx, future - self.s]
        return U @ np.diag(wt) @ V.T


### useful with multi-step or delete?
#    def fit_forecast(self, X, current_time, forecast_step=0):
#        """ Perform RegimeSelection then forecasting

#            X: current tensor
#            current_time: current timepoint
#            forecast_step:
#        """
#        print("fit forecast")
#        _, ridx = self.regime_selection(X, current_time)
#        return self.forecast(ridx, current_time, forecast_step)

    def test(self, X, r_test, output_dir):
        """
            X: a tensor
        """
        n = X.shape[-1]
        Y = np.zeros(X.shape)  # matrix for predictions
        res = []              # array for error measurements  

        print("self.s = " + str(self.s) + ", n = " + str(n) + ", r_test = " + str(r_test))
        for t in trange(self.s, n - r_test, desc='eval', disable=True):
#            Xc = X[..., t-self.s:t]
            Xc = X[..., t-self.s+1:t+1]
            self.update(Xc, t)  # Algorithm 1

            if t % r_test == 0:
                Y[..., t:t+r_test] = self.forecast(
                    self.R[t], t, t, forecast_steps=r_test)
                met = utils.eval(X[..., t:t+r_test], Y[..., t:t+r_test])
                res.append(met)

        print("Total regimes =", self.g)
        print("RMSE =", np.mean(res), " | sd(RMSE) =", np.std(res), 
              " | min(RMSE) =", np.min(res), " | max(RMSE) =", np.max(res))
        print("len(res) = ", len(res))
        print("res = ", res)              
        np.savetxt(output_dir + '/rmse.txt', res)   # rmse per iteration



    def compute_rmse_components(self, X_forecast, X_actual):
        """
        Compute RMSE for all entries, for zeros only, and for non-zero entries.
        
        Args:
            X_forecast (ndarray): Forecasted matrix (2D array).
            X_actual (ndarray): Ground truth matrix (2D array).
        
        Returns:
            rmse_total, rmse_zeros, rmse_nonzeros (floats)
        """
        diff = X_forecast - X_actual
        rmse_total = np.sqrt(np.mean(diff ** 2))
        
        mask_zero = (X_actual == 0)
        mask_nonzero = (X_actual != 0)
        
        if np.sum(mask_zero) > 0:
            rmse_zeros = np.sqrt(np.mean((diff[mask_zero]) ** 2))
        else:
            rmse_zeros = 0.0
        
        if np.sum(mask_nonzero) > 0:
            rmse_nonzeros = np.sqrt(np.mean((diff[mask_nonzero]) ** 2))
        else:
            rmse_nonzeros = 0.0

        return rmse_total, rmse_zeros, rmse_nonzeros        

    def save(self, output_dir):
        np.save(output_dir + '/U.npy', self.U[0])
        np.save(output_dir + '/V.npy', self.U[1])
        np.save(output_dir + '/W.npy', self.W)
        np.savetxt(output_dir + '/R.txt', self.R)   # regime history

    def accumulate_forecasts(self, X):
        """
        Accumulate one-step forecasts and compute RMSE components.
        Returns forecasts and RMSE stats per step.
        """
        n = X.shape[-1]
        forecasts = []
        rmse_total_list = []
        rmse_zeros_list = []
        rmse_nonzeros_list = []

#        for t in trange(self.s, n - 1, desc="Accumulating forecasts"):
        start_t = self.s * self.init_cycles
#        for t in trange(start_t, n - 1, desc="Accumulating forecasts"):
        for t in trange(self.s, n - 1, desc="Accumulating forecasts"):



            Xc = X[..., t-self.s+1:t+1]
            self.update(Xc, t)
            forecast_t = self.forecast(self.R[t], t, steps_ahead=1)
            forecasts.append(forecast_t)

            # Ground truth for next time step
            X_true = X[..., t + 1]
            rmse_total, rmse_zeros, rmse_nonzeros = self.compute_rmse_components(forecast_t, X_true)
            rmse_total_list.append(rmse_total)
            rmse_zeros_list.append(rmse_zeros)
            rmse_nonzeros_list.append(rmse_nonzeros)

        forecasts = np.array(forecasts)

        # Save RMSEs
        np.savetxt(self.output_dir + '/rmse_total_ssmf.txt', np.array(rmse_total_list))
        np.savetxt(self.output_dir + '/rmse_zeros_ssmf.txt', np.array(rmse_zeros_list))
        np.savetxt(self.output_dir + '/rmse_nonzeros_ssmf.txt', np.array(rmse_nonzeros_list))

        print("Average RMSE (total):", np.mean(rmse_total_list))
        print("Average RMSE (zeros):", np.mean(rmse_zeros_list))
        print("Average RMSE (nonzeros):", np.mean(rmse_nonzeros_list))

        return forecasts


if __name__ == '__main__':
    import os
    print("Running from:", os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='disease')
    parser.add_argument('--output_dir', type=str, default='out')
    parser.add_argument('--periodicity', type=int, default=52)
    parser.add_argument('--n_components', type=int, default=10)
    parser.add_argument('--max_regimes', type=int, default=50)
    parser.add_argument('--max_iter', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.2)
    parser.add_argument('--penalty', type=float, default=0.05)
    parser.add_argument('--float_cost', type=int, default=32)
    parser.add_argument('--forecast_step', type=int, default=200)
    parser.add_argument('--update_freq', type=int, default=1)
    config = parser.parse_args()

    utils.make_directory(config.output_dir)

    #tensor = np.load('taxi_yellow_green_rideshare_distinct_march_to_apr2020_tensor.npy')

    tensor = np.load(config.dataset)
    print(f"Data: d1={tensor.shape[0]}, d2={tensor.shape[1]}, T={tensor.shape[2]}")

    model = SSMF(periodicity=config.periodicity,
                 n_components=config.n_components,
                 max_regimes=config.max_regimes,
                 alpha=config.learning_rate,
                 beta=config.penalty,
                 update_freq=config.update_freq,
                 float_cost=config.float_cost)

    model.output_dir = config.output_dir  # add this before model.accumulate_forecasts

    model.initialize(tensor)  # initialize model

    # Uncomment any of your test functions as needed.
    # model.fit(tensor)      
    # model.test(tensor, config.forecast_step, config.output_dir)
    # model.test_per_iteration(tensor, config.output_dir)
    # model.test_with_rmse_components(tensor, 1, config.output_dir)

    # New forecasting accumulation
    forecasts = model.accumulate_forecasts(tensor)
    np.save(config.output_dir + '/ssmf_forecasts.npy', forecasts)
    print("Forecasts accumulated and saved to", config.output_dir + '/ssmf_forecasts.npy')

    model.save(config.output_dir)
    utils.plot_ssmf(config.output_dir, model)
