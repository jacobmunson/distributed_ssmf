import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import preprocessing, metrics
from tqdm import tqdm, trange
from typing import List, Tuple

def make_directory(path, force=True):
    if force:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def read_csv(filename, time_idx):
    df = pd.read_csv(filename)
    df[time_idx] = pd.to_datetime(df[time_idx])
    return df


def list2tensor(df, time_idx, aspects, target, freq):

    date_range = pd.date_range(
        df[time_idx].min(), df[time_idx].max(), freq=freq)

    label_encoders = []

    for col in aspects:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders.append(le)

    # e.g., (n_loc, n_key, duration)
    tensor = np.zeros([len(le.classes_) for le in label_encoders] + [len(date_range)])
    print("Converting to tensor...")

    for key, grouped in tqdm(df.groupby(aspects)):
        tensor[key] = pd.DataFrame(index=date_range).join(
            grouped.set_index(time_idx)[target].resample(freq).sum(),
            how="left").fillna(0).to_numpy().ravel()

    return tensor


def list2tensor_from_index(data, timestamp, n_attributes):
    n_sample = len(timestamp)
    tensor = np.zeros((*n_attributes.values, n_sample))
    print(data)

    # for i, t in enumerate(timestamp):
    for t in trange(n_sample, desc="list2tensor"):
        # print(data[timestamp[t]:timestamp[t+1]].shape)
        if t < n_sample - 1:
            tmp = data[timestamp[t]:timestamp[t+1]]

        if t == n_sample - 1:
            tmp = data[timestamp[t]:]

        for _, row in tmp.iterrows():
            idx, val = row.values[:-1], row.values[-1]
            tensor[idx[0], idx[1], t] += val

    return tensor
    
    
def load_tycho(filename, as_tensor=False):

    # Default setting for TYCHO dataset
    time_idx = "from_date"
    aspects = ["state", "disease"]
    target = "number"
    freq = "W"

    if as_tensor == True:
        if not os.path.isfile("data/project_tycho.npy"):
            data = read_csv(filename, time_idx)
            data = data[data[time_idx] >= "1950-01-01"]
            print(data.nunique())
            tensor = list2tensor(data, time_idx, aspects, target, freq)
            np.save("data/project_tycho.npy", tensor)
            return tensor

        else:
            return np.load("data/project_tycho.npy")

    else:
        return read_csv(filename, time_idx)


#def compute_model_cost(X, n_bits=32, eps=1e-10):
#    k, l = X.shape
#    X_nonzero = np.count_nonzero(X > eps)
#    return X_nonzero * (np.log(k) + np.log(l) + n_bits) + np.log(X_nonzero)

def compute_model_cost(X, n_bits=32, eps=1e-10):
    k, l = X.shape
    X_nonzero = np.count_nonzero(X > eps)

    if X_nonzero == 0:          # ← guard: nothing stored → zero cost
        return 0.0

    return X_nonzero * (np.log(k) + np.log(l) + n_bits) + np.log(X_nonzero)

def coding_cost_tuples(triples, U0, U1, wt, float_cost, d_shape):
    k = len(wt)
    d0, d1 = d_shape

    # Model norm (YY)
    gram0 = U0.T @ U0
    gram1 = U1.T @ U1
    D = np.diag(wt)
    YY = np.trace(gram1 @ D @ gram0 @ D)

    # Data terms
    XY = 0.0
    XX = 0.0
    for r, c, v in triples:
        XX += v * v
        XY += v * (U0[r] * wt @ U1[c])

    err = 0.5 * (YY - 2 * XY + XX)

    # Model bits — revised for parity with dense version
    nz = len(triples)
    model_bits = (
        nz * float_cost +
        np.log(nz + 1e-10) +  # to match log(X_nonzero)
        np.log(d0) +
        np.log(d1)
    )

    return err + model_bits


def compute_coding_cost(X, Y, float_cost=32, masking=True):

    if masking:
        mask = X > 0
        if mask.sum() == 0: return 0
        diff = (X[mask] - Y[mask]).flatten().astype(f'float{float_cost}')

    else:
        diff = (X - Y).flatten().astype(f"float{float_cost}")
    
    logprob = norm.logpdf(diff, loc=diff.mean(), scale=diff.std())
    
    return -1 * logprob.sum() / np.log(2.)

# Add this import at the top of your file
from scipy.stats import norm

def coding_cost_tuples_probabilistic(triples: List[Tuple[int, int, float]],
                                      U0: np.ndarray,
                                      U1: np.ndarray,
                                      wt: np.ndarray,
                                      d_shape: Tuple[int, int]):
    """
    Calculates the MDL cost based on the Gaussian log-likelihood of errors.
    This is the sparse equivalent of the original `compute_coding_cost`.
    """
    if not triples:
        return 0.0

    # 1. Calculate the vector of errors for all non-zero entries
    errors = []
    for r, c, v_true in triples:
        # Reconstruct the predicted value for this one point
        v_pred = np.dot(U0[r], wt * U1[c])
        errors.append(v_true - v_pred)
    
    errors = np.array(errors)

    # We need at least 2 points to calculate a standard deviation
    if errors.size < 2:
        return 0.0
    
    # 2. Fit a Gaussian distribution to the errors
    error_mean = errors.mean()
    error_std = errors.std()

    # To avoid division by zero in logpdf if all errors are identical
    if error_std < 1e-9:
        error_std = 1e-9
    
    # 3. Calculate the total negative log-likelihood in bits
    #    This is the same calculation as in the dense version.
    logprob = norm.logpdf(errors, loc=error_mean, scale=error_std)
    
    # The cost is the number of bits needed to encode the errors.
    # We convert from nats to bits by dividing by log(2).
    return -1 * logprob.sum() / np.log(2.0)


def eval(X, Y):
    mask = X > 0
    return compute_metrics(X[mask], Y[mask])


def compute_metrics(X, Y):
    return np.sqrt(metrics.mean_squared_error(X.ravel(), Y.ravel()))


def plot_ssmf(output_dir, ssmf):
    plt.figure()
    plt.plot(ssmf.R)
    plt.savefig(output_dir + '/R.png')
    plt.close()
    return
