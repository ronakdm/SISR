import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import scipy

from sklearn.preprocessing import LabelEncoder
from scipy.linalg import hilbert, eig
from scipy.stats import boxcox


def create_hilbert_matrix(n, alpha):

    # Get eigenvectors of Hilbert matrix
    H = hilbert(n)
    _, Q = np.linalg.eigh(H)  # Q is orthonormal

    # Create exponentially spaced eigenvalues: lambda_i = exp(alpha * (i - 1)/(n - 1))
    lambdas = np.exp(alpha * np.linspace(0, 1, n))
    Lambda = np.diag(lambdas)

    # Construct matrix with controlled condition number ~ exp(alpha)
    A_alpha = Q @ Lambda @ Q.T
    cond_number = np.linalg.cond(A_alpha)

    return A_alpha, cond_number

def extract_spectrogram_features(x_batch):
    win_len = 50
    window = torch.from_numpy(scipy.signal.windows.hamming(win_len)).float()
    return torch.abs(torchaudio.functional.spectrogram(
        x_batch, 
        0, 
        window, 
        50, 
        win_len // 2, 
        win_len, 
        None, 
        True
    ))

def load_dataset(dataset="laplace_mini", data_path="data/", seed=0, task=None, params={"noise_scale": 0.1, "cond_number": 3}):
    # assert task in ["regression", "classification"]
    if not os.path.exists(data_path):
        raise ValueError(
            f"Invalid 'data_path': '{data_path}'! Please make sure data directory exists."
        )
    data_dir = os.path.join(data_path, dataset)
    if dataset == "laplace_mini":
        C = 10
        T = int(1e3)
        N = 100

        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        A = np.random.normal(size=(C, C))
        signals = []
        for i in range(N):
            signals.append(A @ s[i])
        x = np.stack(signals)
        x -= x.mean(axis=-1)[:, :, None]

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        return x[train_idx], x[test_idx], {"mixing_mat": A}
    elif dataset == "laplace_tall":
        C = 10
        T = int(1e6)
        N = 10
        noise_scale = params["noise_scale"]
        
        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        A = np.random.normal(size=(C, C))
        signals = []
        for i in range(N):
            signals.append((A + noise_scale * np.random.normal(size=A.shape)) @ s[i])
        x = np.stack(signals)
        x -= x.mean(axis=-1)[:, :, None]

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        return x[train_idx], x[test_idx], {"mixing_mat": A}
    elif dataset == "laplace_wide":
        C = 30
        T = int(1e5)
        N = 10

        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        A = np.random.normal(size=(C, C))
        signals = []
        for i in range(N):
            signals.append((A + noise_scale * np.random.normal(size=A.shape)) @ s[i])
        x = np.stack(signals)

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        return x[train_idx], x[test_idx], {"mixing_mat": A}
    elif dataset == "multioutput_simulated":
        C = 10
        T = int(1e3)
        N = 6000
        M = 5
        alpha = params["cond_number"]
        noise_scale = params["noise_scale"]
        A, cond_number = create_hilbert_matrix(C, alpha)
    
        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        n_features = 41 * 26 # n_cells in spectogram for T = 1000
        label_mat = np.random.normal(size=(M, n_features))
        signals = []
        labels = []
        for i in range(N):
            signals.append(A @ s[i])
            h = extract_spectrogram_features(torch.from_numpy(s[i]))
            h = torch.flatten(h, start_dim=1).numpy()

            # only consider the first M sources
            labels.append((label_mat * h[:M]).sum(axis=1))
        x = np.stack(signals)
        labels = np.stack(labels).astype(np.float32) # [N * M]

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        return x_train, y_train, x_test, y_test, {"mixing_mat": A, "label_mat": label_mat}
    elif "reach" in dataset:
       
        # 2000ms contains 500ms rest, 900ms stim, and 600ms rest.
        x = np.load(os.path.join(data_dir, "x.npy")).astype(np.float32)[:, :, :2000]

        N, C, T = x.shape
        np.random.seed(seed)
        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), header=0)
        if task == "regression":
            labels = metadata['reach_time'].to_numpy().astype(np.float32)
            y_train, lmbda = boxcox(labels[train_idx])
            y_test = boxcox(labels[test_idx], lmbda=lmbda)
        elif task == "classification":
            labels = LabelEncoder().fit_transform(metadata['reach_direction'].to_numpy())
            y_train = labels[train_idx]
            y_test = labels[test_idx]
        elif task == "multioutput":
            # reach time
            labels1 = metadata['reach_time'].to_numpy().astype(np.float32)
            labels1_train, lmbda = boxcox(labels1[train_idx])
            labels1_test = boxcox(labels1[test_idx], lmbda=lmbda)

            # planning time
            labels2 = (metadata['planning_end'].to_numpy() - metadata['rest_end'].to_numpy()).astype(np.float32)
            labels2_train, lmbda = boxcox(labels2[train_idx])
            labels2_test = boxcox(labels2[test_idx], lmbda=lmbda)

            # reach direction
            labels3 = LabelEncoder().fit_transform(metadata['reach_direction'].to_numpy())
            labels3_train = labels3[train_idx]
            labels3_test = labels3[test_idx]

            # is stim
            labels4 = LabelEncoder().fit_transform(metadata['is_stim'].to_numpy())
            labels4_train = labels4[train_idx]
            labels4_test = labels4[test_idx]

            y_train = np.stack([labels1_train, labels2_train, labels3_train, labels4_train], axis=1)
            y_test = np.stack([labels1_test, labels2_test, labels3_test, labels4_test], axis=1)

        x_train = x[train_idx]
        x_test = x[test_idx]
        
        return x_train, y_train.astype(np.float32), x_test, y_test.astype(np.float32), metadata
    else:
        raise ValueError(f"No dataset found at {os.path.join(data_path, dataset)}!")
    
    
if __name__ == "__main__":
    for dataset in ["laplace_mini", "laplace_tall", "laplace_wide"]:    
        data, _ = load_dataset(dataset=dataset)
        print(data.shape)
